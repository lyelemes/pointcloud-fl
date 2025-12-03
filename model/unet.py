"""
Dhariwal & Nichol, "Improved Denoising Diffusion Models" (2021).
Paper: https://arxiv.org/abs/2102.09672
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple
import math
import numpy as np

def zero_module(module):
    """
    Sets all parameters of the module to zero and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
    
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim) # to give network more capacity and match dimensions with ResBlock out channels
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.linear1(emb)
        emb = F.silu(emb)
        emb = self.linear2(emb)
        return emb

class DownsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)  # Optional convolution after
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,  time_emb_dim: int = 128):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)  # Project 128 → out_channels
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.bn1(x)
        x = F.silu(x)
        x = self.conv1(x)
        time_projected = self.time_proj(time_emb)  # [B, 128] → [B, out_channels]
        time_projected = time_projected[:, :, None, None]  # [B, out_channels, 1, 1]
        x = x + time_projected  # Add projected time

        x = self.bn2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.skip_connection(residual)

class MultiheadAttention2D(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads)
        self.norm = nn.GroupNorm(32, channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        
        # Reshape: (H*W, B, C) - sequence of pixels
        x_flat = x_norm.view(B, C, H * W).permute(2, 0, 1)
        
        # Self-attention
        attended, _ = self.attention(x_flat, x_flat, x_flat)
        
        # Reshape back: (B, C, H, W)
        attended = attended.permute(1, 2, 0).view(B, C, H, W)
        return x + attended
        
class UNet(nn.Module):
    def __init__(self, in_channels: int, 
                 base_channel: int, 
                 num_res_blocks: int,
                 scheme: list, 
                 resolutions: list, 
                 time_emb_dim: int = 128):
        super(UNet, self).__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)

        encoder_channels = [base_channel] 
        
        self.encoder = [ nn.Conv2d(in_channels, base_channel, kernel_size=3, padding=1)]
        self.decoder = []


        in_channel = base_channel

        res = 1
        for s in scheme:
            for i in range(num_res_blocks):
                out_channel = base_channel * s
                self.encoder.append(ResidualBlock(in_channel, out_channel))
                
                in_channel = out_channel
                encoder_channels.append(in_channel)
                
                if res in resolutions:
                    self.encoder.append(MultiheadAttention2D(out_channel, num_heads=1))

            if s != scheme[-1]:
                self.encoder.append(DownsampleBlock(out_channel))
                res *= 2
                encoder_channels.append(in_channel) # Append again after downsampling for skip connection
        
        self.middle_block = nn.ModuleList([ResidualBlock(in_channel, in_channel), # Bottleneck
                           MultiheadAttention2D(in_channel, num_heads=1),
                           ResidualBlock(in_channel, in_channel)
                           ])
        
        for level_idx, s in enumerate(reversed(scheme)):
            out_channel = base_channel * s
            for i in range(num_res_blocks + 1): # +1 due to downsampling block
                skip_ch = encoder_channels.pop()
                self.decoder.append(ResidualBlock(in_channel+skip_ch, out_channel))
                in_channel = out_channel

                if res in resolutions:
                    self.decoder.append(MultiheadAttention2D(out_channel, num_heads=1))      

            if level_idx < len(scheme) - 1:
                self.decoder.append(UpsampleBlock(out_channel))
                res //= 2
        
        self.out = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=in_channel, eps=1e-6, affine=True),
                           nn.SiLU(),
                           zero_module(nn.Conv2d(in_channel, in_channels, kernel_size=3, padding=1))
                           )
        
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
            
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_embedding(t)
        x_residuals = []
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                x = module(x, t)
                x_residuals.append(x)
            elif isinstance(module, DownsampleBlock):
                x = module(x)
                x_residuals.append(x)
            elif isinstance(module, nn.Conv2d): # Initial conv
                x = module(x)
                x_residuals.append(x)
            else: # Attention
                x = module(x)

        for module in self.middle_block:
            if isinstance(module, ResidualBlock):
                x = module(x, t)
            else: # Attention
                x = module(x)

        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                x = torch.cat((x, x_residuals.pop()), dim=1)
                x = module(x, t)
            elif isinstance(module, UpsampleBlock):
                x = module(x)
            else: # Attention
                x = module(x)

        return self.out(x)
