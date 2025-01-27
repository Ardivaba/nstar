# model.py
import torch.nn as nn
import torch.nn.functional as F

class PathPredictionModel(nn.Module):
    def __init__(self, grid_size=512):
        super().__init__()
        self.grid_size = grid_size
        
        # Enhanced Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        
        # Expanded Bottleneck with Attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        
        # Enhanced Decoder with Skip Connections
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x = x.float()
        
        # Encoder
        x1 = self.enc1(x)   # 512->256
        x2 = self.enc2(x1)  # 256->128
        x3 = self.enc3(x2)  # 128->64
        x4 = self.enc4(x3)  # 64->32
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.dec4(x) + x3  # 32->64
        x = self.dec3(x) + x2  # 64->128
        x = self.dec2(x) + x1  # 128->256
        x = self.dec1(x)       # 256->512
        
        return self.final(x).squeeze(1)