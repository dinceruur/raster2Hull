import fitz
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
import numpy as np
import ot
from tqdm import tqdm
from config.loader import cfg, PROJECT_ROOT
from model.model import ImageToPointCloudModel
from torch.utils.data import DataLoader
from model.dataset_loader import DrawingDataset, drawing_collate_fn, _read_image
import seaborn as sns
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist


def get_ship_hull_cmap(transition_point=0.45):
    colors = [
        (0.0,  "#500000"),  # Deep Dark Red (Keel)
        # Brownish Red (Just below water)
        (transition_point - 0.01, "#A52A2A"),
        # Dark Slate Gray (Just above water)
        (transition_point, "#2F4F4F"),
        (1.0,  "#B0C4DE")   # Light Steel Blue (Deck level)
    ]
    return mcolors.LinearSegmentedColormap.from_list("ShipHull", colors)


def find_nearest_neighbors(pred, gt):
    dists = torch.cdist(pred, gt)
    dists_squared = dists ** 2
    min_dist_p2g, _ = dists_squared.min(dim=2)
    min_dist_g2p, _ = dists_squared.min(dim=1)
    return min_dist_p2g, min_dist_g2p


def calc_chamfer_loss(min_dist_p2g, min_dist_g2p):
    return min_dist_p2g.mean() + min_dist_g2p.mean()


def compute_chamfer_and_fscore(pred_tc, gt_tc, thresholds=[0.01, 0.02, 0.05]):
    dists_p2g, dists_g2p = find_nearest_neighbors(pred_tc, gt_tc)
    cd_val = calc_chamfer_loss(dists_p2g, dists_g2p).item()

    f_scores = {}
    for t in thresholds:
        threshold_sq = t ** 2
        precision = (dists_p2g < threshold_sq).float().mean().item()
        recall = (dists_g2p < threshold_sq).float().mean().item()

        if (precision + recall) > 0:
            f_scores[f"F-Score @ {t*100:.0f}%"] = 2 * (precision * recall) / (precision + recall)
        else:
            f_scores[f"F-Score @ {t*100:.0f}%"] = 0.0

    return cd_val, f_scores


def emd_loss(pred_np, gt_np):
    N = pred_np.shape[0]
    M = gt_np.shape[0]

    # Uniform distributions
    a = np.ones((N,)) / N
    b = np.ones((M,)) / M

    # Cost matrix (squared L2 to match Chamfer scaling)
    C = cdist(pred_np, gt_np, metric='sqeuclidean')
    

    # Exact OT cost
    loss = ot.emd2(a, b, C)

    return loss

def compute_segmentation_metrics(seg_logits, gt_mask, num_classes=4):
    preds = torch.argmax(seg_logits, dim=1).view(-1)
    gt = gt_mask.view(-1)

    # Overall Accuracy
    correct = (preds == gt).sum().item()
    total = gt.numel()
    oa = correct / total if total > 0 else 0.0

    # mIoU calculation
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        gt_inds = (gt == cls)
        intersection = (pred_inds & gt_inds).sum().item()
        union = pred_inds.sum().item() + gt_inds.sum().item() - intersection
        if union > 0:
            iou_list.append(intersection / union)

    miou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    return oa, miou


class Evaluator:
    def __init__(self, model, device, test_loader=None, cfg=None):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.cfg = cfg
        self.model.eval()

    def _npz_path_for(self, identifier):
        """Map a sample identifier (file stem) back to its source .npz path."""
        ds = self.test_loader.dataset
        if not hasattr(self, "_stem2path"):
            self._stem2path = {Path(p).stem: Path(p) for p in ds.file_list}
        return self._stem2path.get(identifier)

    def _load_input_raster(self, identifier, pdf_dir=None, dpi=300):
        ds = self.test_loader.dataset
        prefix = {"ga": "ga_simple"}.get(ds.drawing_type, ds.drawing_type)

        if pdf_dir is not None:
            name = identifier.replace("data_", prefix + "_") + ".pdf"
            for pdf_path in (Path(pdf_dir) / prefix / name, Path(pdf_dir) / name):
                if pdf_path.exists():
                    doc = fitz.open(pdf_path)
                    pix = doc[0].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width)
                    doc.close()
                    return arr.astype(np.float32) / 255.0

        path = self._npz_path_for(identifier)
        if path is None:
            return None
        data = np.load(path, allow_pickle=True)
        img = _read_image(data, ds.keys["image"]).astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        return img

    def visualize_reconstruction_grid(self, pdf_dir=None, num_samples=6, save_path="reconstruction_grid.pdf"):
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap

        if not save_path.lower().endswith('pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'

        self.model.eval()
        print(f"Generating 3D reconstruction grid for {num_samples} samples...")

        # Temporary DataLoader
        temp_loader = DataLoader(
            self.test_loader.dataset,
            batch_size=num_samples,
            shuffle=True,
            collate_fn=self.test_loader.collate_fn
        )
        batch = next(iter(temp_loader))

        # Forward Pass
        with torch.no_grad():
            inputs = batch["images"].to(self.device)
            points_gt = batch["points"].cpu().numpy()
            identifiers = batch["identifiers"]

            predictions, _, _, _, _, _ = self.model(inputs)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            points_pred = predictions["refined"].cpu().numpy()

            imgs_np = inputs.squeeze(1).cpu().numpy()

        # Figure Setup (4 Main Rows)
        fig = plt.figure(figsize=(20, 19), constrained_layout=False)

        # 4 Rows: Image (1), Iso1 (1.5), Iso2 (1.5), Ortho-Block (2.5)
        gs = gridspec.GridSpec(
            4, num_samples, figure=fig, wspace=0.05, hspace=0.05, height_ratios=[1, 1.5, 1.5, 2.5])

        # Color setup
        gt_cmap = get_ship_hull_cmap(transition_point=0.45)
        palette = sns.color_palette("Blues", as_cmap=True)
        pred_color = palette(0.8)

        # Build the Grid
        for i in range(num_samples):

            # Input drawing: full-res raster straight from the .npz, keyed by the
            # active drawing_type (lines / ga_simple / ga_detailed). No dependency
            # on the Tier-2 raw PDFs; pass pdf_dir=<raw/pdf> to prefer those.
            raster = self._load_input_raster(identifiers[i], pdf_dir=pdf_dir)

            ax_img = fig.add_subplot(gs[0, i])
            if raster is not None:
                ax_img.imshow(raster, cmap='gray', vmin=0, vmax=1)
            else:
                print(f"Warning: no raster for {identifiers[i]}. Using model input.")
                ax_img.imshow(imgs_np[i], cmap='gray', vmin=0, vmax=1)

            ax_img.axis('off')
            if i == 0:
                ax_img.set_ylabel("Input\nDrawing",
                                  fontsize=16, fontweight='bold')

            # Point Clouds : Pred & GT
            gt_pc = points_gt[i]
            pr_pc = points_pred[i]

            # Shift to floor (Z=0)
            gt_pc[:, 2] -= gt_pc[:, 2].min()
            pr_pc[:, 2] -= pr_pc[:, 2].min()

            # Mirror Prediction across Y axis
            pr_pc_mirrored = pr_pc.copy()
            pr_pc_mirrored[:, 1] *= -1

            # Normalize Z for GT color mapping
            z = gt_pc[:, 2]
            z_min, z_max = z.min(), z.max()
            z_norm = (z - z_min) / (z_max - z_min + 1e-8)

            # Row 1: iso view
            ax_iso1 = fig.add_subplot(gs[1, i], projection='3d')

            ax_iso1.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2],
                            c=z_norm, cmap=gt_cmap, s=5, alpha=1.0, edgecolors='none')

            ax_iso1.scatter(pr_pc_mirrored[:, 0], pr_pc_mirrored[:, 1], pr_pc_mirrored[:, 2],
                            color=pred_color, s=5, alpha=1.0, edgecolors='none')

            ax_iso1.view_init(elev=20, azim=-45)
            
            for ax_iso in [ax_iso1]:
                ax_iso.set_facecolor('white')
                ax_iso.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax_iso.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax_iso.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax_iso.grid(False)
                ax_iso.axis('off')
                ax_iso.set_xlim(-0.5, 0.5)
                ax_iso.set_ylim(-0.5, 0.5)
                ax_iso.set_zlim(0, 1.0)

            if i == 0:
                ax_iso1.text2D(-0.1, 0.5, "Iso\n(Bow)", transform=ax_iso1.transAxes,
                               fontsize=16, fontweight='bold', ha='center', va='center')

            # Row 2
            ax_iso2 = fig.add_subplot(gs[2, i], projection='3d')

            ax_iso2.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2],
                          c=z_norm, cmap=gt_cmap, s=5, alpha=1.0, edgecolors='none')
            ax_iso2.scatter(pr_pc_mirrored[:, 0], pr_pc_mirrored[:, 1], pr_pc_mirrored[:, 2],
                            color=pred_color, s=5, alpha=1.0, edgecolors='none')

            ax_iso2.view_init(elev=30, azim=135) 
            
            for ax_iso in [ax_iso2]:
                ax_iso.set_facecolor('white')
                ax_iso.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax_iso.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax_iso.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax_iso.grid(False)
                ax_iso.axis('off')
                ax_iso.set_xlim(-0.5, 0.5)
                ax_iso.set_ylim(-0.5, 0.5)
                ax_iso.set_zlim(0, 1.0)

            if i == 0:
                ax_iso2.text2D(-0.1, 0.5, "Iso\n(Stern)", transform=ax_iso2.transAxes,
                               fontsize=16, fontweight='bold', ha='center', va='center')


            # ROW 3
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=gs[3, i], hspace=0.1)

            def plot_2d_projection(inner_idx, x_gt, y_gt, x_pr, y_pr, label, is_first_col):
                ax = fig.add_subplot(inner_gs[inner_idx])
                ax.scatter(x_gt, y_gt, c=z_norm, cmap=gt_cmap,
                           s=5, alpha=1.0, edgecolors='none')
                ax.scatter(x_pr, y_pr, color=pred_color,
                           s=5, alpha=1.0, edgecolors='none')
                ax.axis('equal')
                ax.axis('off')
                if is_first_col:
                    ax.text(-0.1, 0.5, label, transform=ax.transAxes,
                            fontsize=14, fontweight='bold', ha='center', va='center')

            # Profile (Side): X vs Z
            plot_2d_projection(0, gt_pc[:, 0], gt_pc[:, 2], pr_pc_mirrored[:, 0],
                               pr_pc_mirrored[:, 2], "Profile\n(Side)", i == 0)

            # Body (Front): Y vs Z
            plot_2d_projection(1, gt_pc[:, 1], gt_pc[:, 2], pr_pc_mirrored[:, 1],
                               pr_pc_mirrored[:, 2], "Body\n(Front)", i == 0)

            # Top (Half-Breadth): X vs Y
            plot_2d_projection(2, gt_pc[:, 0], gt_pc[:, 1], pr_pc_mirrored[:, 0],
                               pr_pc_mirrored[:, 1], "Top\n(Half-Breadth)", i == 0)

        # Legend
        import matplotlib.lines as mlines
        gt_marker = mlines.Line2D([], [], color='#A52A2A', marker='o', linestyle='None',
                                  markersize=10, label='Ground Truth (Starboard / Custom Paint)')
        pr_marker = mlines.Line2D([], [], color=pred_color, marker='o', linestyle='None',
                                  markersize=10, label='Prediction (Port / Mirrored)')

        fig.legend(handles=[gt_marker, pr_marker], loc='upper center', bbox_to_anchor=(0.5, 0.95),
                   ncol=2, fontsize=16, frameon=False)

        plt.show()
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Done! Saved perfectly formatted 3D Grid to {save_path}")

    def visualize_segmentation(self, pdf_dir=None, num_samples=4, save_path="seg_qualitative_highres.pdf"):
        if not save_path.lower().endswith('pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'

        self.model.eval()
        print(
            f"Sampling {num_samples} random images for high-res PDF visualization...")

        # Create a temporary dataloader
        temp_loader = DataLoader(
            self.test_loader.dataset,
            batch_size=num_samples,
            shuffle=True,
            collate_fn=self.test_loader.collate_fn
        )

        batch = next(iter(temp_loader))

        # Setup 2x2 Grid with tight margins
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={
                                 'wspace': 0.05, 'hspace': 0.05})
        axes_flat = axes.flatten()

        # Extract and reorder seaborn 'tab10' palette
        tab10 = sns.color_palette()
        alpha = 0.4
        cmap_colors = [
            (0.0, 0.0, 0.0, 0.0),       # Class 0: Background (Transparent)
            (*tab10[0], alpha),         # Class 1: XZ / Sheer (Blue)
            (*tab10[3], alpha),         # Class 2: YZ / Body (Red)
            (*tab10[2], alpha),         # Class 3: XY / Half-Breadth (Green)
        ]
        custom_cmap = ListedColormap(cmap_colors)

        with torch.no_grad():
            inputs = batch["images"].to(self.device)
            identifiers = batch["identifiers"]  # ['data_6087_0', ...]

            _, _, _, _, seg_logits, _ = self.model(inputs)

            # Predictions
            seg_preds_base = torch.argmax(
                seg_logits, dim=1).float()  # (B, H, W)

            for i in range(num_samples):
                ax = axes_flat[i]

                # Input drawing: full-res raster from the .npz, keyed by the active
                # drawing_type. Falls back to a Tier-2 raw PDF only if pdf_dir is
                # given and the matching file exists.
                raster = self._load_input_raster(identifiers[i], pdf_dir=pdf_dir)
                if raster is None:
                    print(f"Warning: no raster for {identifiers[i]}. Skipping.")
                    continue
                H_hr, W_hr = raster.shape[:2]

                # Upsample prediction mask to the raster resolution
                mask_tensor = seg_preds_base[i].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                mask_high_res = F.interpolate(
                    mask_tensor, size=(H_hr, W_hr), mode='nearest'
                ).squeeze().cpu().numpy()

                # Plot drawing + segmentation overlay
                ax.imshow(raster, cmap='gray', vmin=0, vmax=1)
                ax.imshow(mask_high_res, cmap=custom_cmap, interpolation='nearest')

                ax.axis('off')

        # Legend
        patches_list = [
            patches.Patch(color=(*tab10[0], alpha), label='Sheer Plan (XZ)'),
            patches.Patch(color=(*tab10[3], alpha), label='Body Plan (YZ)'),
            patches.Patch(color=(*tab10[2], alpha), label='Half-Breadth (XY)')
        ]

        fig.legend(handles=patches_list, loc='lower center', bbox_to_anchor=(0.5, 0.05),
                   ncol=3, fontsize=14, frameon=False)

        plt.savefig(save_path, format="pdf", dpi=400, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Done! Saved ultra-high-res visualization to {save_path}")

    def _generate_seg_mask(self, seg_logits, bboxes, bbox_classes):
        B, n_classes, Hf, Wf = seg_logits.shape
        device = seg_logits.device
        gt_mask = torch.zeros((B, Hf, Wf), dtype=torch.long, device=device)
        for b in range(B):
            valid_indices = torch.nonzero(bbox_classes[b]).squeeze(1)
            for v_idx in valid_indices:
                cls = bbox_classes[b, v_idx].item()
                bx1, by1, bx2, by2 = bboxes[b, v_idx]
                x1 = max(0, int(bx1 * Wf))
                y1 = max(0, int(by1 * Hf))
                x2 = min(Wf, int(torch.ceil(bx2 * Wf)))
                y2 = min(Hf, int(torch.ceil(by2 * Hf)))
                region = gt_mask[b, y1:y2, x1:x2]
                region[region == 0] = cls
        return gt_mask

    @torch.no_grad()
    def evaluate_section_on_test_set(self, fscore_threshold=0.01):
        if self.test_loader is None:
            raise ValueError(
                "Test loader must be provided for bulk evaluation.")
 
        # Per-sample lists for mean & std computation
        all_cd = []
        all_emd = []
        all_norm_cd = []
        all_norm_emd = []
        all_fscores = {f"F-Score @ {t*100:.0f}%": [] for t in [0.01, 0.02, 0.05]}
 
        all_sectional_f1 = {
            "Stern F1 @ 1%":   [],
            "Midship F1 @ 1%": [],
            "Bow F1 @ 1%":     [],
        }
 
        # Segmentation
        all_oa = []
        all_miou = []
 
        print(f"Starting Inference on Test Dataset (F-Score Threshold: {fscore_threshold})...")
 
        for batch in tqdm(self.test_loader):
            # Unpack batch
            inputs = batch["images"].to(self.device)
            points_gt = batch["points"].to(self.device)
            bboxes = batch["bboxes_coordinates"].to(self.device)
            bbox_classes = batch["bbox_classes"].to(self.device)
 
            # Forward pass
            predictions, F1, view_weights, final_weights, seg_logits, view_type_flat = self.model(inputs)
 
            if isinstance(predictions, tuple):
                points_pred_dict = predictions[0]
            else:
                points_pred_dict = predictions
 
            points_pred = points_pred_dict["refined"]
            B = points_gt.shape[0]
 
            # Per-sample metrics (segmentation + point cloud)
            gt_seg_mask = self._generate_seg_mask(seg_logits, bboxes, bbox_classes)
 
            for i in range(B):
                oa, miou = compute_segmentation_metrics(
                    seg_logits[i:i+1], gt_seg_mask[i:i+1], num_classes=4
                )
                all_oa.append(oa)
                all_miou.append(miou)
 
            for i in range(B):
                p_pred = points_pred[i:i+1]
                p_gt = points_gt[i:i+1]
 
                # Global metrics
                cd_val, f_score_dict = compute_chamfer_and_fscore(
                    p_pred, p_gt, thresholds=[0.01, 0.02, 0.05]
                )
 
                # POT metric EMD
                p_pred_np = p_pred.squeeze(0).cpu().numpy()
                p_gt_np = p_gt.squeeze(0).cpu().numpy()
                emd_val = emd_loss(p_pred_np, p_gt_np)
 
                # Bounding Box Diagonal (Squared) for Normalization
                min_pt = p_gt_np.min(axis=0)
                max_pt = p_gt_np.max(axis=0)
                diagonal_sq = np.sum((max_pt - min_pt) ** 2)
 
                norm_cd  = (cd_val  / diagonal_sq) if diagonal_sq > 0 else 0.0
                norm_emd = (emd_val / diagonal_sq) if diagonal_sq > 0 else 0.0
 
                all_cd.append(cd_val)
                all_emd.append(emd_val)
                all_norm_cd.append(norm_cd)
                all_norm_emd.append(norm_emd)
                for k, v in f_score_dict.items():
                    all_fscores[k].append(v)
 
                # Sectional evaluation
                boundary_stern = -0.7
                boundary_bow   =  0.7
 
                mask_gt_stern  = p_gt[:, :, 0] < boundary_stern
                mask_gt_mid    = (p_gt[:, :, 0] >= boundary_stern) & (p_gt[:, :, 0] <= boundary_bow)
                mask_gt_bow    = p_gt[:, :, 0] > boundary_bow
 
                mask_pred_stern = p_pred[:, :, 0] < boundary_stern
                mask_pred_mid   = (p_pred[:, :, 0] >= boundary_stern) & (p_pred[:, :, 0] <= boundary_bow)
                mask_pred_bow   = p_pred[:, :, 0] > boundary_bow
 
                gt_stern   = p_gt[mask_gt_stern].unsqueeze(0)
                gt_mid     = p_gt[mask_gt_mid].unsqueeze(0)
                gt_bow     = p_gt[mask_gt_bow].unsqueeze(0)
                pred_stern = p_pred[mask_pred_stern].unsqueeze(0)
                pred_mid   = p_pred[mask_pred_mid].unsqueeze(0)
                pred_bow   = p_pred[mask_pred_bow].unsqueeze(0)
 
                if gt_stern.shape[1] > 0 and pred_stern.shape[1] > 0:
                    _, f_stern = compute_chamfer_and_fscore(pred_stern, gt_stern, thresholds=[0.01])
                    all_sectional_f1["Stern F1 @ 1%"].append(f_stern["F-Score @ 1%"])
 
                if gt_mid.shape[1] > 0 and pred_mid.shape[1] > 0:
                    _, f_mid = compute_chamfer_and_fscore(pred_mid, gt_mid, thresholds=[0.01])
                    all_sectional_f1["Midship F1 @ 1%"].append(f_mid["F-Score @ 1%"])
 
                if gt_bow.shape[1] > 0 and pred_bow.shape[1] > 0:
                    _, f_bow = compute_chamfer_and_fscore(pred_bow, gt_bow, thresholds=[0.01])
                    all_sectional_f1["Bow F1 @ 1%"].append(f_bow["F-Score @ 1%"])
 
        # Compute Mean & Std
        def stats(lst):
            a = np.array(lst)
            return a.mean(), a.std()
 
        num_samples = len(all_cd)
        cd_mean,   cd_std   = stats(all_cd)
        emd_mean,  emd_std  = stats(all_emd)
        ncd_mean,  ncd_std  = stats(all_norm_cd)
        nemd_mean, nemd_std = stats(all_norm_emd)
        oa_mean,   oa_std   = stats(all_oa)
        miou_mean, miou_std = stats(all_miou)
 
        fscore_stats   = {k: stats(v) for k, v in all_fscores.items()}
        sectional_stats = {k: stats(v) for k, v in all_sectional_f1.items() if len(v) > 0}
 
        # Print Results
        W = 55
        print("\n" + "=" * W)
        print("TEST SET EVALUATION RESULTS  (mean ± std)")
        print("=" * W)
        print(f"Total Samples Tested  : {num_samples}")
        print(f"Raw CD  (×10³)        : {cd_mean*1000:.3f} ± {cd_std*1000:.3f}")
        print(f"Raw EMD (×10³)        : {emd_mean*1000:.3f} ± {emd_std*1000:.3f}")
        print(f"Norm CD  Error        : {ncd_mean*100:.2f}% ± {ncd_std*100:.2f}%")
        print(f"Norm EMD Error        : {nemd_mean*100:.2f}% ± {nemd_std*100:.2f}%")
        print("-" * W)
        print("GLOBAL F-SCORES:")
        for k, (m, s) in fscore_stats.items():
            print(f"  {k:<20}: {m*100:.2f}% ± {s*100:.2f}%")
        print("-" * W)
        print("SECTIONAL F-SCORES:")
        for k, (m, s) in sectional_stats.items():
            print(f"  {k:<20}: {m*100:.2f}% ± {s*100:.2f}%")
        print("-" * W)
        print(f"Segmentation mIoU     : {miou_mean*100:.2f}% ± {miou_std*100:.2f}%")
        print(f"Segmentation OA       : {oa_mean*100:.2f}% ± {oa_std*100:.2f}%")
        print("=" * W)
 
        # Package Results Dictionary  (mean + std for every metric)
        results = {
            "cd_10e3_mean":      cd_mean * 1000,   "cd_10e3_std":      cd_std * 1000,
            "emd_10e3_mean":     emd_mean * 1000,  "emd_10e3_std":     emd_std * 1000,
            "norm_cd_pct_mean":  ncd_mean * 100,   "norm_cd_pct_std":  ncd_std * 100,
            "norm_emd_pct_mean": nemd_mean * 100,  "norm_emd_pct_std": nemd_std * 100,
            "miou_pct_mean":     miou_mean * 100,  "miou_pct_std":     miou_std * 100,
            "oa_pct_mean":       oa_mean * 100,    "oa_pct_std":       oa_std * 100,
        }
 
        for k, (m, s) in fscore_stats.items():
            ck = k.lower().replace(" ", "_").replace("@_", "").replace("%", "_pct")
            results[f"{ck}_mean"] = m * 100
            results[f"{ck}_std"]  = s * 100
 
        for k, (m, s) in sectional_stats.items():
            ck = k.lower().replace(" ", "_").replace("@_", "").replace("%", "_pct")
            results[f"{ck}_mean"] = m * 100
            results[f"{ck}_std"]  = s * 100
 
        return results


if __name__ == "__main__":
    # Config
    train_cfg = cfg.training

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ImageToPointCloudModel(cfg=cfg.model).to(device)

    # Checkpoint
    checkpoint_path = str(
        PROJECT_ROOT / train_cfg.checkpoints_dir / "checkpoint_latest.pth.tar")

    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Setup DataLoader
    test_loader = DataLoader(
        DrawingDataset(cfg.dataset, split="test"),
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=drawing_collate_fn,
    )

    # Instantiate Evaluator
    evaluator = Evaluator(model, device, test_loader=test_loader, cfg=cfg)

    RUN_MODE = "both"

    evaluator.visualize_reconstruction_grid(num_samples=4)

    if RUN_MODE in ["visualize_seg", "both"]:
        # Runs instantly, picks random files, saves the picture, and finishes.
        evaluator.visualize_segmentation(
            num_samples=4,
            save_path="seg_qualitative.pdf"
        )

    if RUN_MODE in ["evaluate", "both"]:
        # Metrics over the entire dataset.
        metrics = evaluator.evaluate_section_on_test_set(fscore_threshold=0.01)