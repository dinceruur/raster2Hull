import torch
import torch.nn as nn
import torch.nn.functional as F
from model.conv_block import ResBlock, Stem
from torchvision.ops import SqueezeExcitation

class Encoder(nn.Module):

    def __init__(self, input_channel=1, stem_channel=32, stage_channels=[64, 128, 256]):
        super().__init__()
        # c1=64 (Fine), c2=128 (Mid), c3=256 (Global)
        c1, c2, c3 = stage_channels 

        # Stem (H -> H)
        # Stride 1 convolution to capture raw edge topology without crushing it.
        self.stem = Stem(input_channel, stem_channel)

        # Stage 1: H -> H/4 (Total Stride 4)
        self.layer1 = nn.Sequential(
            # Step A: H -> H/2
            # Expand channels (32->64)
            ResBlock(stem_channel, c1, stride=2, use_se=False), 
            
            # Step B: Preserve H/2
            # Processes features at H/2 before the next downsample happens.
            ResBlock(c1, c1, stride=1, use_se=False), 
            
            # Step C: H/2 -> H/4
            ResBlock(c1, c1, stride=2, use_se=False)  
        )

        # Stage 2: H/4 -> H/8 (Total Stride 2)
        self.layer2 = nn.Sequential(
            # Step A: Preserve H/4
            ResBlock(c1, c1, stride=1, use_se=True), 
            
            # Step B: H/4 -> H/8 (Increase channels 64->128)
            ResBlock(c1, c2, stride=2, use_se=True),
            
            # Step C: Deepen at H/8
            ResBlock(c2, c2, stride=1, use_se=True)  
        )

        # Stage 3: H/8 -> H/16 (Total Stride 2)
        self.layer3 = nn.Sequential(
            # Step A: H/8 -> H/16 (Increase channels 128->256)
            ResBlock(c2, c3, stride=2, use_se=True),
            
            # Step B: Deep Processing
            ResBlock(c3, c3, stride=1, use_se=True),
            ResBlock(c3, c3, stride=1, use_se=True) 
        )

       
        # Top-Down Pathway
        # Projects high-level semantics down to low-level features.
        
        # Lateral connections (1x1 convs) to match channel dimensions
        self.lat_layer3 = nn.Conv2d(c3, c2, 1) # 256 -> 128
        self.lat_layer2 = nn.Conv2d(c2, c1, 1) # 128 -> 64

        # Smooth layers (3x3) to remove aliasing artifacts from upsampling
        self.smooth2 = nn.Conv2d(c2, c2, 3, padding=1)
        self.smooth1 = nn.Conv2d(c1, c1, 3, padding=1)

    
        # Gating Mechanism
        # A learnable mask:
        # 0 = Use Local only (Sharp lines)
        # 1 = Use Context (Fill gaps)
        self.gate1 = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self):
        # Standard initialization for the main spatial pathway
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Override for SE Blocks (Neutral Gating Initialization)
        for m in self.modules():
            if isinstance(m, SqueezeExcitation):
                # Init with a small normal distribution.
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Conv2d):
                        nn.init.normal_(sub_m.weight, mean=0.0, std=0.01)
                        if sub_m.bias is not None:
                            nn.init.zeros_(sub_m.bias)

        # Override for FPN Gating Mechanism
        # Ensure the FPN gate also starts neutral (outputting ~0.5 via Sigmoid)
        nn.init.normal_(self.gate1[0].weight, mean=0.0, std=0.01)
        if self.gate1[0].bias is not None:
            # Start with a strong bias towards the local pathway
            nn.init.constant_(self.gate1[0].bias, -2.0)

    def forward(self, x):
        
        # Bottom-Up (Encoding)
        x = self.stem(x)        # H
        c1 = self.layer1(x)     # H/4
        c2 = self.layer2(c1)    # H/8
        c3 = self.layer3(c2)    # H/16

        # Top-Down (Decoding/Fusion)

        # Fuse P3 (H/16) into P2 (H/8)
        p3 = self.lat_layer3(c3)
        p3_up = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth2(c2 + p3_up)           

        # Fuse P2 (H/8) into P1 (H/4)
        p2_proj = self.lat_layer2(p2)
        p2_up = F.interpolate(p2_proj, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        
        # Gated Injection at H/4
        # Let the network decide where to add
        gate = self.gate1(c1) 
        p1 = c1 + (p2_up * gate) 
        
        # Final Smoothing
        F1_final = self.smooth1(p1)
        
        return F1_final