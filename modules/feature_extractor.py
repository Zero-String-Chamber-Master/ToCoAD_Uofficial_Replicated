# modules/feature_extractor.py
import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """
    返回 ResNet / WideResNet 多层特征：
    - f2: layer2 输出
    - f3: layer3 输出
    - f4: layer4 输出
    ToCoAD / DRAEM 风格的 decoder 通常用多层特征做 skip-connection。
    """

    def __init__(self, name: str = "wide_resnet50_2",
                 pretrained: bool = True,
                 frozen: bool = True):
        super().__init__()
        if name == "wide_resnet50_2":
            net = models.wide_resnet50_2(pretrained=pretrained)
        elif name == "resnet18":
            net = models.resnet18(pretrained=pretrained)
        elif name == "resnet50":
            net = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # 1/4

        x1 = self.layer1(x)   # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        return x2, x3, x4     # f2, f3, f4
