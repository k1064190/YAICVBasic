import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # img shape: (1, 572, 572)
        # contracting path
        self.down1 = DoubleConv(3, 64)  # 64x256x256
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 64x128x128
        self.down2 = DoubleConv(64, 128)    # 128x128x128
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 128x64x64
        self.down3 = DoubleConv(128, 256)   # 256x64x64
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)   # 256x32x32
        self.down4 = DoubleConv(256, 512)   # 512x32x32
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)   # 512x16x16
        self.down5 = DoubleConv(512, 1024)  # 1024x16x16

        # expansive path
        self.unpool4 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)    # 512x32x32
        self.up4 = DoubleConv(1024, 512)    # 512x32x32
        self.unpool3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)    # 256x64x64
        self.up3 = DoubleConv(512, 256)     # 256x64x64
        self.unpool2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)    # 128x128x128
        self.up2 = DoubleConv(256, 128)     # 128x128x128
        self.unpool1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)     # 64x256x256
        self.up1 = DoubleConv(128, 64)      # 64x256x256

        self.out = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=False)  # num_classesx256x256



    def forward(self, x):
        layer_after_conv1 = self.down1(x)
        layer_after_conv2 = self.maxpool1(layer_after_conv1)
        layer_after_conv2 = self.down2(layer_after_conv2)
        layer_after_conv3 = self.maxpool2(layer_after_conv2)
        layer_after_conv3 = self.down3(layer_after_conv3)
        layer_after_conv4 = self.maxpool3(layer_after_conv3)
        layer_after_conv4 = self.down4(layer_after_conv4)
        layer_after_conv5 = self.maxpool4(layer_after_conv4)
        layer_after_conv5 = self.down5(layer_after_conv5)

        layer_after_upconv4 = self.unpool4(layer_after_conv5)
        layer_after_upconv4 = torch.cat([layer_after_conv4 ,layer_after_upconv4], dim=1)
        layer_after_upconv4 = self.up4(layer_after_upconv4)
        layer_after_upconv3 = self.unpool3(layer_after_upconv4)
        layer_after_upconv3 = torch.cat([layer_after_conv3, layer_after_upconv3], dim=1)
        layer_after_upconv3 = self.up3(layer_after_upconv3)
        layer_after_upconv2 = self.unpool2(layer_after_upconv3)
        layer_after_upconv2 = torch.cat([layer_after_conv2, layer_after_upconv2], dim=1)
        layer_after_upconv2 = self.up2(layer_after_upconv2)
        layer_after_upconv1 = self.unpool1(layer_after_upconv2)
        layer_after_upconv1 = torch.cat([layer_after_conv1, layer_after_upconv1], dim=1)
        layer_after_upconv1 = self.up1(layer_after_upconv1)
        out = self.out(layer_after_upconv1)
        return out



