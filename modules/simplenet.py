# modules/simplenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_backbone


class FeatureAdapter(nn.Module):
    """1x1 conv + BN + ReLU，把 ImageNet 特征适配到目标域。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class AnomalyFeatureGenerator(nn.Module):
    """在特征空间加高斯噪声，生成伪异常特征。"""

    def __init__(self, sigma: float = 0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # 推理时不再使用
            return feat
        noise = torch.randn_like(feat) * self.sigma
        return feat + noise


class PatchDiscriminator(nn.Module):
    """
    简单 patch-level 判别器：1x1 conv -> logit map [B,1,H,W]，
    输出值越大代表越“异常”。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x  # raw logits


class SimpleNet(nn.Module):
    """
    SimpleNet 纯 PyTorch 实现：
    - Feature Extractor: ResNet / WideResNet
    - Feature Adapter: 1x1 conv
    - Anomaly Feature Generator: add Gaussian noise
    - Anomaly Discriminator: 1x1 conv patch classifier
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        feature_dim: int = 256,
        noise_sigma: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.encoder, enc_channels = get_backbone(
            name=backbone_name,
            pretrained=True,
            frozen=freeze_backbone,
        )
        self.adapter = FeatureAdapter(enc_channels, feature_dim)
        self.generator = AnomalyFeatureGenerator(sigma=noise_sigma)
        self.discriminator = PatchDiscriminator(feature_dim)

    # ----------- 拆成几个小步骤，训练 / 推理都能用 ----------- #
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()  # backbone 始终 eval
        feat = self.encoder(x)
        return feat

    def adapt(self, feat: torch.Tensor) -> torch.Tensor:
        return self.adapter(feat)

    def generate_anomaly_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return self.generator(feat)

    def discriminate(self, feat: torch.Tensor) -> torch.Tensor:
        """返回 patch-level logits [B,1,H,W]"""
        return self.discriminator(feat)

    # ----------- 推理：输出 anomaly map ----------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理用：输入图像，输出 anomaly map (0-1) 大小为特征图尺度。
        """
        with torch.no_grad():
            f = self.encode(x)
        f = self.adapt(f)
        logits = self.discriminate(f)
        prob = torch.sigmoid(logits)
        return prob
