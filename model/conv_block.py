import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation


def gn(num_channels, max_groups=8):
    return nn.GroupNorm(
        num_groups=min(max_groups, num_channels),
        num_channels=num_channels
    )


class AntiAliasSkip(nn.Module):

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        
        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

        x = self.avg_pool(x)
        return self.conv(x)


class Stem(nn.Module):

    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            # Expansion
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=3,
                      padding=1, bias=False),
            gn(out_ch // 2),
            nn.SiLU(inplace=True),

            # Refinement
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3,
                      padding=1, bias=False),
            gn(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.use_se = use_se

        self.gn1 = gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.gn2 = gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)

        if self.use_se:
            self.se = SqueezeExcitation(
                input_channels=out_ch,
                squeeze_channels=max(1, out_ch // 16),
                activation=nn.SiLU,
                scale_activation=nn.Sigmoid
            )

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            if stride > 1:
                self.downsample = AntiAliasSkip(in_ch, out_ch, stride=stride)
            else:
                self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        # Pre-activate the input
        preact = F.silu(self.gn1(x), inplace=False)

        # Main Convolutional Path
        out = self.conv1(preact)
        
        # Second Pre-activation
        out = self.conv2(F.silu(self.gn2(out), inplace=True))

        if self.use_se:
            out = self.se(out)

        # Skip Connection
        if self.downsample is not None:
            identity = self.downsample(preact)
        else:
            identity = x

        # Pure Addition
        return out + identity