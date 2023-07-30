import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
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
        self.down1 = DoubleConv(1, 64)  # 572x572x1 -> 570x570x64 -> 568x568x64
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 568x568x64 -> 284x284x64
        self.down2 = DoubleConv(64, 128)    # 284x284x64 -> 282x282x128 -> 280x280x128
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 280x280x128 -> 140x140x128
        self.down3 = DoubleConv(128, 256)   # 140x140x128 -> 138x138x256 -> 136x136x256
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)   # 136x136x256 -> 68x68x256
        self.down4 = DoubleConv(256, 512)   # 68x68x256 -> 66x66x512 -> 64x64x512
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)   # 64x64x512 -> 32x32x512
        self.down5 = DoubleConv(512, 1024)  # 32x32x512 -> 30x30x1024 -> 28x28x1024

        # expansive path
        self.unpool4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, bias=False)    # 28x28x1024 -> 56x56x512
        self.up4 = DoubleConv(1024, 512)    # 56x56x512 -> 54x54x512 -> 52x52x512
        self.unpool3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=False)    # 52x52x512 -> 104x104x256
        self.up3 = DoubleConv(512, 256)     # 104x104x512 -> 102x102x256 -> 100x100x256
        self.unpool2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False)    # 100x100x256 -> 200x200x128
        self.up2 = DoubleConv(256, 128)     # 200x200x256 -> 198x198x128 -> 196x196x128
        self.unpool1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False)     # 196x196x128 -> 392x392x64
        self.up1 = DoubleConv(128, 64)      # 392x392x128 -> 390x390x64 -> 388x388x64

        self.out = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=1, bias=False)  # 388x388x64 -> 388x388x2



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
        layer_after_upconv4 = torch.cat([transforms.CenterCrop((layer_after_upconv4.shape(2),
                                              layer_after_upconv4.shape(3)))(layer_after_conv4), layer_after_upconv4], dim=1)
        layer_after_upconv4 = self.up4(layer_after_upconv4)
        layer_after_upconv3 = self.unpool3(layer_after_upconv4)
        layer_after_upconv3 = torch.cat([transforms.CenterCrop((layer_after_upconv3.shape(2),
                                              layer_after_upconv3.shape(3)))(layer_after_conv3), layer_after_upconv3], dim=1)
        layer_after_upconv3 = self.up3(layer_after_upconv3)
        layer_after_upconv2 = self.unpool2(layer_after_upconv3)
        layer_after_upconv2 = torch.cat([transforms.CenterCrop((layer_after_upconv2.shape(2),
                                              layer_after_upconv2.shape(3)))(layer_after_conv2), layer_after_upconv2], dim=1)
        layer_after_upconv2 = self.up2(layer_after_upconv2)
        layer_after_upconv1 = self.unpool1(layer_after_upconv2)
        layer_after_upconv1 = torch.cat([transforms.CenterCrop((layer_after_upconv1.shape(2),
                                              layer_after_upconv1.shape(3)))(layer_after_conv1), layer_after_upconv1], dim=1)
        layer_after_upconv1 = self.up1(layer_after_upconv1)
        out = self.out(layer_after_upconv1)
        return out



