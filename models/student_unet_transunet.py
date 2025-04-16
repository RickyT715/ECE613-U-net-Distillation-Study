import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class StudentTransUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StudentTransUNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)  # 112x112

        self.encoder2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 56x56

        self.bridge = DoubleConv(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 56 -> 112
        self.decoder1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 112 -> 224
        self.decoder2 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.pool1(x1)

        x3 = self.encoder2(x2)
        x4 = self.pool2(x3)

        x5 = self.bridge(x4)

        x6 = self.up1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.decoder1(x6)

        x7 = self.up2(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.decoder2(x7)

        return self.out_conv(x7)
