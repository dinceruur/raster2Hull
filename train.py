import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict


from config.loader import cfg, PROJECT_ROOT
from model.loss import TotalLoss
from model.model import ImageToPointCloudModel

from model.dataset_loader import DrawingDataset, drawing_collate_fn
from model.utility import save_checkpoint, load_checkpoint, count_params


VIEW_CLASS_MAP = {
    "YZ": 1,  # Body plan or mid-section
    "XY": 2,  # Half-breadth or top-view
    "XZ": 3,  # Sheer plan or profile view
}

# Visualization & Logging Helper
class TensorboardVisualizer:

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def close(self):
        self.writer.close()

    @staticmethod
    def to_cpu(x):
        return x.detach().cpu() if torch.is_tensor(x) else x

    @staticmethod
    def sanitize_for_logging(x, eps=1e-6):
        if x is None:
            return None
        x = x.detach()
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def log_pointcloud(self, tag, points, step):
        self.writer.add_mesh(
            tag=tag, vertices=points.unsqueeze(0), global_step=step
        )

    def log_point_view_weights(self, view_weights, step, tag="view_weights", num_points=5):
        w = view_weights[0].detach().cpu()
        for i in range(min(num_points, w.shape[0])):
            self.writer.add_histogram(f"{tag}/point_{i}", w[i], step)

    def log_view_usage(self, view_weights, step, tag="view_usage"):
        w = view_weights[0].detach().cpu()
        usage = w.mean(dim=0)
        for v in range(usage.shape[0]):
            self.writer.add_scalar(f"{tag}/view_{v}", usage[v].item(), step)

    def log_cross_attention(self, final_weights, F1, original_img, step, tag="attention", query_indices=(0, 100, 500), threshold=0.10):
        attn = final_weights[0].detach().cpu()
        if attn.dim() == 3:
            attn = attn.mean(dim=0)

        _, _, Hf, Wf = F1.shape
        img = original_img[0].cpu().numpy()
        if img.ndim == 3:
            img = img[0]

        H_img, W_img = img.shape
        img_rgb = np.stack([img] * 3, axis=-1)
        img_rgb = (img_rgb * 255).clip(0, 255).astype(np.uint8)

        def apply_thresholded_heatmap(attention_matrix):
            attn_grid = attention_matrix.view(Hf, Wf)
            map_resized = F.interpolate(
                attn_grid[None, None], size=(H_img, W_img), mode="nearest"
            ).squeeze().numpy()

            map_norm = map_resized / (map_resized.max() + 1e-8)
            heatmap = cv2.applyColorMap(
                np.uint8(255 * map_norm), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = img_rgb.copy()
            mask = map_norm > threshold
            overlay[mask] = (img_rgb[mask] * 0.4 +
                             heatmap[mask] * 0.6).astype(np.uint8)
            return overlay

        global_attn = attn.mean(dim=0)
        self.writer.add_image(
            f"{tag}/global_mean", apply_thresholded_heatmap(global_attn), step, dataformats="HWC")

        for q_idx in query_indices:
            if q_idx < attn.shape[0]:
                self.writer.add_image(
                    f"{tag}/query_{q_idx}", apply_thresholded_heatmap(attn[q_idx]), step, dataformats="HWC")

    def log_segmentation_logits(self, seg_logits, original_img, step, tag="train_segmentation/overlay"):
        img = original_img[0].detach().cpu().permute(1, 2, 0).numpy()
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        img_rgb = np.repeat(
            img_uint8, 3, axis=2) if img_uint8.shape[2] == 1 else img_uint8
        H, W = img_rgb.shape[:2]

        pred_indices = torch.argmax(seg_logits[0:1], dim=1).detach().cpu()
        pred_indices_resized = F.interpolate(
            pred_indices.unsqueeze(1).float(), size=(H, W), mode="nearest"
        ).squeeze().long().numpy()

        colors = np.array([
            [0, 0, 0],    # Class 0: Background
            [255, 0, 0],  # Class 1: YZ
            [0, 0, 255],  # Class 2: XY
            [0, 255, 0],  # Class 3: XZ
        ], dtype=np.uint8)

        color_mask = colors[pred_indices_resized]
        is_foreground = pred_indices_resized > 0
        overlay = img_rgb.copy()

        blended_full = cv2.addWeighted(img_rgb, 0.6, color_mask, 0.4, 0)
        overlay[is_foreground] = blended_full[is_foreground]

        self.writer.add_image(tag, overlay, step, dataformats="HWC")

    def log_layout_maps(self, layout, step, tag_prefix, max_images=1):
        B, K, H, W = layout.shape
        layout = layout[:max_images]
        for k in range(K):
            layout_k = layout[:, k: k + 1].clamp(0, 1)
            grid = vutils.make_grid(layout_k, nrow=max_images, normalize=True)
            self.writer.add_image(
                f"{tag_prefix}/layout_{k}", grid, global_step=step)


class PointCloudTrainer:
    """Manages data loading, model initialization, and the training loop."""

    def __init__(self, config):
        self.cfg = config
        self.train_cfg = self.cfg.training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        # Paths
        self.checkpoints_dir = (PROJECT_ROOT / self.train_cfg.checkpoints_dir).resolve()
        print(f"Checkpoints directory: {self.checkpoints_dir}")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Vis
        self.visualizer = TensorboardVisualizer(log_dir=str(self.checkpoints_dir))

        # Tracking
        self.start_epoch = 0

    def setup(self):
        """Initializes datasets, model, optimizer, and loads checkpoints."""
        self._setup_dataloaders()
        self._setup_model()
        

    def _setup_dataloaders(self):
        num_workers = 4

        self.train_loader = DataLoader(
            DrawingDataset(self.cfg.dataset, split="train"),
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=drawing_collate_fn,
        )

        self.val_loader = DataLoader(
            DrawingDataset(self.cfg.dataset, split="val"),
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=drawing_collate_fn,
        )

        self.test_loader = DataLoader(
            DrawingDataset(self.cfg.dataset, split="test"),
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=drawing_collate_fn,
        )

    def _setup_model(self):
        self.model = ImageToPointCloudModel(cfg=self.cfg.model).to(self.device)
        self.model.print_parameter_counts()

        loss_config = {
            "recon_type": "weighted_chamfer",   # "chamfer", "weighted_chamfer", "emd"
            "normal_type": "weighted_normal",   # "none", "normal", "weighted_normal"
            "use_range": True,                  # Range Penalty ON
            "use_seg": True,                    # Segmentation ON
            "use_view_coverage" : True,
            "use_spatial_div" : True
        }
        
        self.criterion = TotalLoss(exp_config=loss_config).to(self.device)

        print(self.criterion.config)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.train_cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_cfg.num_epochs, eta_min=1e-5
        )

        self.resumed_best_recon = float('inf')
        self.resumed_no_improve = 0

        if self.train_cfg.resume_training:
            resume_path = (PROJECT_ROOT / self.train_cfg.resume_checkpoint_path).resolve()
            print(resume_path)
            if resume_path.exists():
                self.start_epoch, _, self.resumed_best_recon, self.resumed_no_improve = load_checkpoint(
                    resume_path, self.model, self.optimizer, self.scheduler
                )
                print(f"Resumed from epoch {self.start_epoch} | best_recon={self.resumed_best_recon:.4f} | no_improve={self.resumed_no_improve}")
            else:
                print("Checkpoint not found. Starting fresh.")
                

    
    def train_one_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        loss_tracker = defaultdict(float)

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.train_cfg.num_epochs}")
        # last_batch_data = {}


        for batch_idx, batch in enumerate(loop):
            # Move to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "identifiers"}

            self.optimizer.zero_grad()

            pred_points, F1, view_weights, final_weights, seg_logits, view_type_flat = self.model(
                inputs["images"], inputs["bboxes_coordinates"], inputs["bbox_masks"], inputs["bbox_classes"]
            )
            
            losses = self.criterion(
                points_pred=pred_points,
                seg_logits=seg_logits,
                view_weights=view_weights,
                final_weights=final_weights, 
                view_type_flat=view_type_flat,
                points_gt=inputs["points"], 
                normals_gt=inputs["normals"],
                bboxes=inputs["bboxes_coordinates"], 
                bbox_classes=inputs["bbox_classes"],
                epoch=epoch, 
            )
            
            losses["total_loss"].backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            for key, val in losses.items():
                loss_tracker[key] += val.item()

            loop.set_postfix(loss=f"{(loss_tracker['total_loss'] / (batch_idx + 1)):.4f}")


        self.scheduler.step()
        self.visualizer.writer.add_scalar(
            "LR", self.scheduler.get_last_lr()[0], epoch
        )

        # Log average training losses
        num_batches = len(self.train_loader)
        avg_train_losses = {} # Store averages safely

        for key, total_val in loss_tracker.items():
            avg_loss = total_val / num_batches
            avg_train_losses[key] = avg_loss
            self.visualizer.writer.add_scalar(f"Loss/Train/{key}", avg_loss, epoch)


        return avg_train_losses

    def validate(self, epoch):
        self.model.eval()
        self.criterion.eval()
        val_loss_tracker = defaultdict(float)

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {k: v.to(self.device)
                          for k, v in batch.items() if k != "identifiers"}

                pred_points, F1, view_weights, final_weights, seg_logits, view_type_flat = self.model(
                    inputs["images"], inputs["bboxes_coordinates"], inputs["bbox_masks"], inputs["bbox_classes"]
                )

                losses = self.criterion(
                    points_pred=pred_points,
                    seg_logits=seg_logits,
                    view_weights=view_weights,
                    final_weights=final_weights,
                    view_type_flat=view_type_flat,
                    points_gt=inputs["points"],
                    normals_gt=inputs["normals"],
                    bboxes=inputs["bboxes_coordinates"],
                    bbox_classes=inputs["bbox_classes"],
                    epoch=epoch,
                )

                for key, val in losses.items():
                    val_loss_tracker[key] += val.item()

        # Dynamic averaging and logging
        num_batches = len(self.val_loader)
        avg_val_losses = {}

        for key, total_val in val_loss_tracker.items():
            avg_loss = total_val / num_batches
            avg_val_losses[key] = avg_loss
            self.visualizer.writer.add_scalar(f"Loss/Val/{key}", avg_loss, epoch)

        return avg_val_losses

    def export_and_save(self, epoch, train_losses, val_losses):
        if (epoch + 1) % int(self.train_cfg.save_every) != 0:
            return

        # Checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "train_loss": train_losses["total_loss"],
                "val_loss": val_losses["total_loss"],
            },
            checkpoint_dir=self.checkpoints_dir,
            filename=f"checkpoint_epoch_{epoch+1}.pth.tar",
        )

    def run(self):
        print("Starting training...")
        self.setup()

        # Early Stopping Initialization
        best_recon_loss = self.resumed_best_recon
        epochs_no_improve = self.resumed_no_improve

        # Get patience from config
        early_stop_patience = getattr(self.train_cfg, 'early_stop_patience', 75) 

        for epoch in range(self.start_epoch, self.train_cfg.num_epochs):
            train_losses = self.train_one_epoch(epoch)
            val_losses = self.validate(epoch)

            # Extract reconstruction loss
            val_recon = val_losses.get('loss_recon', float('inf'))

            # Print the main reconstruction and total losses
            loss_str = (
                f"Train Total: {train_losses.get('total_loss', 0):.4f} | "
                f"Val Total: {val_losses.get('total_loss', 0):.4f}"
            )
            print(f"\nEpoch {epoch+1} | {loss_str}")

            # Periodic export, standard checkpointing (save_every)
            self.export_and_save(epoch, train_losses, val_losses)

            # Overwrite latest checkpoint for pause-and-resume
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "train_loss": train_losses["total_loss"],
                    "val_loss": val_losses["total_loss"],
                    "best_recon_loss": best_recon_loss,
                    "epochs_no_improve": epochs_no_improve,
                },
                checkpoint_dir=self.checkpoints_dir,
                filename="checkpoint_latest.pth.tar",
            )

            # Early stopping, best model checkpointing
            # Epochs 0-50 are skipped entirely: no saving, no counting
            if epoch > 50:
                if val_recon < best_recon_loss:
                    best_recon_loss = val_recon
                    epochs_no_improve = 0
                    print(f"--> New Best Model! Validation Recon: {best_recon_loss:.4f}. Saving 'checkpoint_best.pth.tar'...")

                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "train_loss": train_losses["total_loss"],
                            "val_loss": val_losses["total_loss"],
                            "val_recon": val_recon,
                            "best_recon_loss": best_recon_loss,
                            "epochs_no_improve": epochs_no_improve,
                        },
                        checkpoint_dir=self.checkpoints_dir,
                        filename="checkpoint_best.pth.tar",
                    )
                else:
                    epochs_no_improve += 1
                    print(f"No improvement. Early Stopping Counter: {epochs_no_improve}/{early_stop_patience}")

                    if epochs_no_improve >= early_stop_patience:
                        print(f"Terminating Training: Validation Recon has not improved for {early_stop_patience} epochs.")
                        break

        self.visualizer.close()


if __name__ == "__main__":
    trainer = PointCloudTrainer(config=cfg)
    trainer.run()
