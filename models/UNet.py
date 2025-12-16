import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.silu(x)
        
        return x + identity

class UNet(nn.Module):
    def __init__(self,cfg, in_channels=1, out_channels=1, time_dim=128):
        super().__init__()
        
        # time emb
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        
        # time proj
        self.time_proj1 = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.SiLU()
        )
        self.time_proj2 = nn.Sequential(
            nn.Linear(time_dim, 256),
            nn.SiLU()
        )
        
        # input conv
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # down 1
        self.down1_blocks = nn.ModuleList([
            ResidualBlock(64, 64),
        ])
        self.down1_conv = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        
        # down 2
        self.down2_blocks = nn.ModuleList([
            ResidualBlock(128, 128),
        ])
        self.down2_conv = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        
        # bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(256, 256),
        ])
        
        # upsample 1
        self.up1_conv = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up1_blocks = nn.ModuleList([
            ResidualBlock(256, 128),  # 256 due to skip connection
        ])
        
        # upsample 2
        self.up2_conv = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2_blocks = nn.ModuleList([
            ResidualBlock(128, 64),  # 128 due to skip connection
        ])
        
        # output conv
        self.conv_out = nn.Sequential(
            ResidualBlock(64, 64),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
    def forward(self, x, t):

        # get time embedding
        t = t.float()
        t = self.time_mlp(t.view(-1, 1))
        
        # conv1 
        x = self.conv_in(x)
        skip1 = x # set skip 1
        
        # downsample1 + add timeproj1
        for block in self.down1_blocks:
            x = block(x)
        x = self.down1_conv(x)
        x = x + self.time_proj1(t).view(-1, 128, 1, 1)
        skip2 = x # set skip 2
        
        #downsample2 + add timeproj2
        for block in self.down2_blocks:
            x = block(x)
        x = self.down2_conv(x)
        x = x + self.time_proj2(t).view(-1, 256, 1, 1)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x)
            
        # upsample1 + concat skip2
        x = self.up1_conv(x)
        x = torch.cat([x, skip2], dim=1)
        for block in self.up1_blocks:
            x = block(x)
            
        # upsample2 + concat skip1
        x = self.up2_conv(x)
        x = torch.cat([x, skip1], dim=1)
        for block in self.up2_blocks:
            x = block(x)
            
        # output conv
        x = self.conv_out(x)
        
        return x