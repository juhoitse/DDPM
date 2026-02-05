import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from blocks import ResidualBlock, Downsample, PositionalEmbedding, Upsample

class UNet(nn.Module):
    """
    U-Net architecture for DDPM.
    Conditions on both timestep and class label.
    """
    def __init__(self, img_channels = 1, base_channels = 32, 
                 num_classes = None, time_emb_dim = None):
        super().__init__()
        
        self.num_classes = num_classes

        self.time_emb_dim = time_emb_dim
        
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None


        # encoder
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.enc11 = ResidualBlock(base_channels, base_channels, time_emb_dim, num_classes)
        self.down1 = Downsample(base_channels)
        self.enc21 = ResidualBlock(base_channels, 2*base_channels, time_emb_dim, num_classes)
        self.down2 = Downsample(2*base_channels)
        self.enc31 = ResidualBlock(2*base_channels, 2*base_channels, time_emb_dim, num_classes)
        self.down3 = Downsample(2*base_channels)
        self.enc41 = ResidualBlock(2*base_channels, 2*base_channels, time_emb_dim, num_classes)
        
        self.encoder_layers = [
            self.enc11, self.down1,
            self.enc21, self.down2,
            self.enc31, self.down3,
            self.enc41,
        ]

        # bottleneck
        self.mid1 = ResidualBlock(2*base_channels, 2*base_channels, time_emb_dim, num_classes)
        
        # decoder
        self.dec11 = ResidualBlock(4*base_channels, 2*base_channels, time_emb_dim, num_classes)
        self.dec13 = ResidualBlock(4*base_channels, 2*base_channels, time_emb_dim, num_classes)
        self.up1 = Upsample(2*base_channels)

        self.dec21 = ResidualBlock(4*base_channels, 2*base_channels, time_emb_dim, num_classes)
        self.dec23 = ResidualBlock(4*base_channels, 2*base_channels, time_emb_dim, num_classes)
        self.up2 = Upsample(2*base_channels)

        self.dec31 = ResidualBlock(4*base_channels, 2*base_channels, time_emb_dim, num_classes)
        self.dec33 = ResidualBlock(3*base_channels, base_channels, time_emb_dim, num_classes)
        self.up3 = Upsample(base_channels)

        self.dec41 = ResidualBlock(2*base_channels, base_channels, time_emb_dim, num_classes)
        self.dec43 = ResidualBlock(2*base_channels, base_channels, time_emb_dim, num_classes)
        
        self.decoder_layers = [
            self.dec11, self.dec13, self.up1,
            self.dec21, self.dec23, self.up2,
            self.dec31, self.dec33, self.up3,
            self.dec41, self.dec43,
        ]

        self.out_conv = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
        
    
    def forward(self, x, time=None, labels = None):
        """Forward pass for the UNet
        Args:
        x of shape (batch_size, n_channels, H, W): Images 
        time of shape (batch_size,): Timesteps for embedding, None if no conditioning on time
        labels of shape (batch_size,): Classes of the images, None if no conditioning on class
        """
        
        if self.time_emb_dim is not None:
            time_emb = self.time_mlp(time)
        else:
            time_emb = None

        skips = []
        out = self.init_conv(x)
        skips.append(out)
        # go through the encoder layers and save skips from every layer, including upsamples
        for layer in self.encoder_layers:
            out = layer(out, time_emb, labels)
            skips.append(out)

        out = self.mid1(out, time_emb, labels)
        
        # do through the decoder layers, skips are given to residual blocks only
        for i, layer in enumerate(self.decoder_layers):
            if i != 2 and i != 5 and i != 8:
                skip = skips.pop()
                out = torch.cat([out, skip], dim = 1)
            out = layer(out, time_emb, labels) 
        out = self.out_conv(out)
        
        return out

