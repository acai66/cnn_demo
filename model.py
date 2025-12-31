import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """极简的多层卷积神经网络.

    只有卷积、正则化、激活函数，最后一层是全局平均池化，中间没有池化层，整个网络不含全连接层。
    正则化在部署时可以合并到卷积层中。激活函数运算过程非常简单，部署时可以当成卷积层的一部分。
    当输入分辨率为32时，全局平均池化等效于被忽略.
    可以通过修改各层的 self.conv_1_channels、self.conv_2_channels、self.conv_3_channels、self.conv_4_channels 参数调整整个网络的参数量、运算量
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        self.conv_1_channels = 16
        self.conv_1 = nn.Conv2d(
            input_channels, self.conv_1_channels, kernel_size=3, stride=2, padding=1
        )
        self.bn_1 = nn.BatchNorm2d(self.conv_1_channels)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv_2_channels = 24
        self.conv_2 = nn.Conv2d(
            self.conv_1_channels,
            self.conv_2_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.bn_2 = nn.BatchNorm2d(self.conv_2_channels)
        self.relu_2 = nn.ReLU(inplace=True)

        self.conv_3_channels = 32
        self.conv_3 = nn.Conv2d(
            self.conv_2_channels,
            self.conv_3_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.bn_3 = nn.BatchNorm2d(self.conv_3_channels)
        self.relu_3 = nn.ReLU(inplace=True)

        self.conv_4_channels = 48
        self.conv_4 = nn.Conv2d(
            self.conv_3_channels,
            self.conv_4_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.bn_4 = nn.BatchNorm2d(self.conv_4_channels)
        self.relu_4 = nn.ReLU(inplace=True)

        self.classifier = nn.Conv2d(
            self.conv_4_channels,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)

        x = self.classifier(x)
        x = self.avg_pool(x)

        return x.flatten(1)
