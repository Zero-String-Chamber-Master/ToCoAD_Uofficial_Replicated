# modules/discriminative_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DiscriminativeUNet(nn.Module):
    """
    简化版 U-Net 风格 decoder：
    输入：多层特征 (f2, f3, f4)
    输出：2 通道 mask logits [B,2,H,W]（H,W 接近 f2 尺度）
    """

    def __init__(self,
                 ch2: int,
                 ch3: int,
                 ch4: int,
                 base_ch: int = 256):
        super().__init__()

        # 把不同层通道统一到 base_ch
        self.l2_conv = nn.Conv2d(ch2, base_ch, kernel_size=1)
        self.l3_conv = nn.Conv2d(ch3, base_ch, kernel_size=1)
        self.l4_conv = nn.Conv2d(ch4, base_ch, kernel_size=1)

        # decoder blocks
        self.dec4 = ConvBlock(base_ch, base_ch)
        self.dec3 = ConvBlock(base_ch * 2, base_ch)  # concat f3
        self.dec2 = ConvBlock(base_ch * 2, base_ch)  # concat f2

        self.out_conv = nn.Conv2d(base_ch, 2, kernel_size=1)  # 2 类：normal / anomaly

    def forward(self, feats):
        """
        feats: (f2, f3, f4)
        f2: [B,c2,H/8,W/8]
        f3: [B,c3,H/16,W/16]
        f4: [B,c4,H/32,W/32]
        """
        f2, f3, f4 = feats

        f2_ = self.l2_conv(f2)  # [B,base,H/8,W/8]
        f3_ = self.l3_conv(f3)  # [B,base,H/16,W/16]
        f4_ = self.l4_conv(f4)  # [B,base,H/32,W/32]

        # 从最深层开始解码
        x = self.dec4(f4_)  # [B,base,H/32,W/32]

        # up -> f3 尺度
        x = F.interpolate(x, size=f3_.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, f3_], dim=1)  # [B,2*base,H/16,W/16]
        x = self.dec3(x)                # [B,base,H/16,W/16]

        # up -> f2 尺度
        x = F.interpolate(x, size=f2_.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, f2_], dim=1)  # [B,2*base,H/8,W/8]
        x = self.dec2(x)                # [B,base,H/8,W/8]

        logits = self.out_conv(x)       # [B,2,H/8,W/8]
        return logits
