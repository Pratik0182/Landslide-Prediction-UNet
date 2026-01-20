import torch
import torch.nn as nn

#define the u-net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sequence(x)

class UNet(nn.Module):
    def __init__(self, in_channels=14, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3 = DoubleConv(256 + 512, 256)
        self.up2 = DoubleConv(128 + 256, 128)
        self.up1 = DoubleConv(64 + 128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        x = self.pool(c1)
        c2 = self.down2(x)
        x = self.pool(c2)
        c3 = self.down3(x)
        x = self.pool(c3)
        x = self.down4(x)
        x = self.upsample(x)
        x = torch.cat([x, c3], dim=1)
        x = self.up3(x)
        x = self.upsample(x)
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, c1], dim=1)
        x = self.up1(x)
        return torch.sigmoid(self.final_conv(x))
