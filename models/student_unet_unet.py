import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class StudentUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StudentUNet, self).__init__()

        self.encoder1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)  # 输出尺寸: (B, 32, 112, 112)

        self.encoder2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 输出尺寸: (B, 64, 56, 56)

        self.decoder1 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  #(B, 32, 112, 112)

        self.decoder2 = DoubleConv(96, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  #(B, 16, 224, 224)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)  #(B, 32, 224, 224)
        x2 = self.pool1(x1)  #(B, 32, 112, 112)

        x3 = self.encoder2(x2)  #(B, 64, 112, 112)
        x4 = self.pool2(x3)  #(B, 64, 56, 56)

        x5 = self.decoder1(x4)  #(B, 32, 56, 56)
        x5 = self.up1(x5)  #(B, 32, 112, 112)

        #保证维度匹配
        if x5.size(2) != x3.size(2) or x5.size(3) != x3.size(3):
            x5 = F.interpolate(x5, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=False)

        x6 = torch.cat([x5, x3], dim=1)  # (B, 96, 112, 112)
        x7 = self.decoder2(x6)  # (B, 32, 112, 112)
        x7 = self.up2(x7)  # (B, 16, 224, 224)

        x8 = self.final_conv(x7)  # (B, out_channels, 224, 224)

        #保证输出输入匹配
        x8 = F.interpolate(x8, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x8
