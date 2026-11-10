import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Encoder
from model.light_seg import SegHeadWithContext
from model.mean_shape import MeanShapeTemplate
from model.decoder import PointRefiner


class ImageToPointCloudModel(nn.Module):
    """
    Pipeline:
        Image
          ↓
        Encoder → F1: Spatial Features
          ↓
        Segmentation Head
          ↓
        MeanShapeTemplate → Coarse Point Cloud Prior
          ↓
        PointRefiner (Dynamic Tokenizer & Cross-Attention)
          ↓
        Refined Point Cloud
    """

    def __init__(self, cfg):
        super().__init__()

        # Configuration unpacking
        input_channels = cfg.input_channels
        stem_channels = cfg.stem_channels
        enc_stage_channels = cfg.enc_channels
        num_views = cfg.num_views

        attn_hidden_dim = cfg.attn_hidden_dim
        attn_head = cfg.attn_head
        attn_layers = cfg.attn_layers
        
        # Image Encoder
        # Produces: (B, C1, H/4, W/4)
        self.encoder = Encoder(
            input_channel=input_channels,
            stem_channel=stem_channels,
            stage_channels=enc_stage_channels
        )
       
        # View / Layout Segmentation Head
        # Predicts per-pixel view class:
        #   0 = background
        #   1..num_views = semantic views (XY, YZ, XZ)
        # Operates on F1 to preserve spatial precision.
        self.seg_head = SegHeadWithContext(
            in_chans=enc_stage_channels[0],
            num_views=num_views,
            hidden_dim=64
        )

        self.template_module = MeanShapeTemplate(
            mean_shape_path="mean_ship.npy",
            num_points=1024,
            jitter_factor=0.02  # 2% of the standard deviation
        )

        # Point Refinement Transformer
        self.point_refiner = PointRefiner(
            feat_dim=enc_stage_channels[0],  # F1 channel dimension
            hidden_dim=attn_hidden_dim,
            n_head=attn_head,
            num_layers=attn_layers,
            num_views=num_views,
        )

    def print_parameter_counts(self):
        print(f"\n{'='*40}")
        print(f"{'Module':<20} | {'Trainable Params':<15}")
        print(f"{'-'*20}-+-{'-'*15}")

        total_params = 0
        for name, child in self.named_children():
            count = sum(p.numel()
                        for p in child.parameters() if p.requires_grad)
            print(f"{name:<20} | {count:>13,}")
            total_params += count

        print(f"{'-'*20}-+-{'-'*15}")
        print(f"{'TOTAL':<20} | {total_params:>13,}")
        print(f"{'='*40}\n")

    def forward(self, img, bboxes=None, bbox_mask=None, bbox_classes=None):
        points = {}

        # Encode image
        F1 = self.encoder(img)

        # Mean shape
        B = F1.shape[0]   # batch size from encoder output

        # Predict view/layout segmentation
        # Shape: (B, num_views+1, Hf, Wf)
        seg_logits = self.seg_head(F1)

        # Template Shape
        template = self.template_module(B, apply_jitter=self.training)

        # Transformer Decoder
        p_refined, view_weights, final_weights, view_type_flat, delta = self.point_refiner(
            F1=F1,
            P=template,
            seg_logits=seg_logits,
            bboxes=bboxes,
            bbox_mask=bbox_mask,
            bbox_classes=bbox_classes
        )

        points["template"] = template
        points["delta"] = delta
        points["refined"] = p_refined

        # Outputs
        return points, F1, view_weights, final_weights, seg_logits, view_type_flat
