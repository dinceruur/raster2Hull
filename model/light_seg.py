import torch
import torch.nn as nn
import torch.nn.functional as F


class SegHeadWithContext(nn.Module):
   
    def __init__(self, in_chans, num_views, hidden_dim=64):
        super().__init__()

        # Bottleneck (Reduce channels)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Global Context Branch
        # Squeezes the whole image to 1x1 vector to capture layout
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_process = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial Processing (Standard 3x3)
        self.process = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Prediction
        self.pred = nn.Conv2d(hidden_dim, num_views + 1, kernel_size=1)

    def forward(self, x):
        # x: (B, in_chans, H, W)

        # Reduce channels (B, 64, H, W)
        x = self.reduce(x)

        # Calculate Global Context (B, 64, 1, 1)
        global_ctx = self.global_pool(x)
        global_ctx = self.global_process(global_ctx)

        # Inject Context: 
        x = x * global_ctx

        # Spatial refinement
        x = self.process(x)

        return self.pred(x)
