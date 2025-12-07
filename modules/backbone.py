# modules/backbone.py
import torch
import torch.nn as nn
from torchvision import models


def get_backbone(name: str = "wide_resnet50_2",
                 pretrained: bool = True,
                 frozen: bool = True) -> tuple[nn.Module, int]:
    """
    返回特征提取 backbone，只保留到最后一个 conv block 的特征图。
    输出: (encoder, out_channels)
    """
    if name == "wide_resnet50_2":
        backbone = models.wide_resnet50_2(pretrained=pretrained)
    elif name == "resnet18":
        backbone = models.resnet18(pretrained=pretrained)
    elif name == "resnet50":
        backbone = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    # 去掉 avgpool 和 fc，只保留到 layer4 的 conv
    modules_list = [
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    ]
    encoder = nn.Sequential(*modules_list)
    out_channels = backbone.layer4[-1].conv3.out_channels if hasattr(backbone.layer4[-1], "conv3") \
        else backbone.layer4[-1].conv2.out_channels

    if frozen:
        for p in encoder.parameters():
            p.requires_grad = False

    return encoder, out_channels
