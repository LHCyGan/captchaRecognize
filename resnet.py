# -*- encoding:utf-8 -*-
# author: liuheng
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

class ResBlock(nn.Module):
    """resnet block"""
    def __init__(self, channels_in, channels_out):
        """
        :param channels_in: 输入通道数
        :param channels_out: 输出通道数
        """
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels_out)

        self.extra = nn.Sequential()
        if channels_out != channels_in:
            self.extra = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(channels_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.extra(x) + out
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer2 = ResBlock(64, 64)
        self.layer3 = ResBlock(64, 128)
        self.layer4 = ResBlock(128, 256)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


if __name__ == '__main__':
    x = torch.ones((1, 3) + (64, 160))
    net = ResNet18()
    print(net(x).shape)



