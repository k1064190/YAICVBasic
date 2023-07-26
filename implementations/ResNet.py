import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # BasicBlock의 경우 expansion = 1
        expansion = 1
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*ResidualBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*ResidualBlock.expansion)
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*ResidualBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*ResidualBlock.expansion)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # Bottleneck의 경우 expansion = 4로 두어 마지막 layer의 output channel을 4배로 늘려줌
        expansion = 4
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels*Bottleneck.expansion)
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels*Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*Bottleneck.expansion)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, init_weights=True):
        self.in_channels = 64
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = self._make_multiple_blocks(block, num_blocks[0], 64)
        self.conv3 = self._make_multiple_blocks(block, num_blocks[1], 128, stride=2)
        self.conv4 = self._make_multiple_blocks(block, num_blocks[2], 256, stride=2)
        self.conv5 = self._make_multiple_blocks(block, num_blocks[3], 512, stride=2)

        self.avg_pool = nn.AvgPool2d(kernel_size=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Conv를 kaiming normal로 초기화하고 bias를 0으로 초기화,
        # Linear와 BatchNorm의 weights는 각각 0.01, 1로 초기화 후 bias는 0으로 초기화
        if init_weights:
            self._initialize_weights()

    def _make_multiple_blocks(self, block, num_blocks, out_channels, stride=1):
        blocks = []
        for i in range(num_blocks):
            blocks.append(block(self.in_channels, out_channels, stride if i == 0 else 1))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):   # conv2d의 weight를 kaiming normal로 초기화
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None: # bias가 존재한다면 0으로 초기화
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avg_pool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(ResidualBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

