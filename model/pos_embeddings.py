import math
import torch
from torch import nn


class PositionEmbeddingSine2D(nn.Module):
    """
    DETR-style sine positional encoding for 2D feature maps.
    Normalized to [0, 2π].
    Reference : https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    """

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, H, W, device):
        # Create grid
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H - 1 + eps) * self.scale
            x_embed = x_embed / (W - 1 + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, device=device, dtype=torch.float32
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        # (H, W, C)
        return torch.cat([pos_y, pos_x], dim=2)


class PositionEmbeddingSine3D(nn.Module):
    """
    Modified DETR-style sine positional embedding for 3D point clouds.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        
        if scale is not None:
            self.scale = scale
        elif not normalize:
            # If inputs are raw [-1, 1], scale them to be "pixel-like"
            self.scale = 128 * 2 * math.pi
        else:
            self.scale = 2 * math.pi

    def forward(self, xyz):
        assert xyz.dim() == 3 and xyz.size(-1) == 3
        x, y, z = xyz.unbind(dim=-1)

        if self.normalize:
            # Only use 'normalize=True' if your inputs are NOT naturally in [-1,1].
            eps = 1e-6
            x = (x - x.min(dim=1, keepdim=True)[0]) / (
                x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + eps)
            y = (y - y.min(dim=1, keepdim=True)[0]) / (
                y.max(dim=1, keepdim=True)[0] - y.min(dim=1, keepdim=True)[0] + eps)
            z = (z - z.min(dim=1, keepdim=True)[0]) / (
                z.max(dim=1, keepdim=True)[0] - z.min(dim=1, keepdim=True)[0] + eps)
            
            # For normalized inputs [0,1], 2pi scale is standard DETR behavior
            scale = 2 * math.pi
        else:
            # Use the larger scale
            scale = self.scale

        x = x * scale
        y = y * scale
        z = z * scale

        dim_t = torch.arange(
            self.num_pos_feats, device=xyz.device, dtype=torch.float32
        )
        # dim_t ranges from 1 to 10000
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        def encode(pos):
            # pos is (B, N)
            # dim_t is (C,)
            # result: (B, N, C)
            pos = pos.unsqueeze(-1) / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()),
                dim=3
            ).flatten(2)
            return pos

        pos_x = encode(x)
        pos_y = encode(y)
        pos_z = encode(z)

        return torch.cat([pos_x, pos_y, pos_z], dim=-1)
