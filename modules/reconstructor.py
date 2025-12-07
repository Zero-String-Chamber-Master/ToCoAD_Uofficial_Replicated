# modules_stage2_tocoad/reconstructor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """简单的 Conv-BN-ReLU 模块"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpBlock(nn.Module):
    """
    上采样 + skip 连接 + ConvBlock
    in_ch: 上采样后特征的通道数
    skip_ch: skip 特征的通道数
    out_ch: 输出通道数
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: [B,in_ch,h,w], skip: [B,skip_ch,H,W]
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ReconstructionUNet(nn.Module):
    """
    使用 F 的多层特征 (f2,f3,f4) 做 U-Net 解码，重构回 3 通道图像。
    约定：
      - f2: [B,C2,H/4,W/4]
      - f3: [B,C3,H/8,W/8]
      - f4: [B,C4,H/16,W/16]
    最终输出分辨率为 H/4 × W/4。
    """

    def __init__(
        self,
        ch2: int,
        ch3: int,
        ch4: int,
        base_ch: int = 256,
        out_channels: int = 3,
    ):
        super().__init__()

        # bottleneck：先把 f4 -> 4*B
        self.block4 = nn.Sequential(
            ConvBlock(ch4, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4),
        )

        # 上采样 + skip
        self.up3 = UpBlock(in_ch=base_ch * 4, skip_ch=ch3, out_ch=base_ch * 2)
        self.up2 = UpBlock(in_ch=base_ch * 2, skip_ch=ch2, out_ch=base_ch)

        # 输出层：恢复到 3 通道
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, feats):
        """
        feats: (f2, f3, f4)
        返回：x_rec [B,3,H/4,W/4]
        """
        f2, f3, f4 = feats

        x = self.block4(f4)      # [B,4B,H/16,W/16]
        x = self.up3(x, f3)      # [B,2B,H/8,W/8]
        x = self.up2(x, f2)      # [B,B,H/4,W/4]
        x = self.out_conv(x)     # [B,3,H/4,W/4]
        return x
