# modules_stage2_5/stage2_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2FeatureDiffHead(nn.Module):
    """
    基于 F 的多层特征做差分，输出一张 anomaly logit map：
    - 输入（两种调用方式都支持）：
        方式 A：位置参数
            forward(f2_orig, f3_orig, f4_orig, f2_rec, f3_rec, f4_rec)

        方式 B：关键字参数
            forward(feats_orig=(f2_orig, f3_orig, f4_orig),
                    feats_rec=(f2_rec,  f3_rec,  f4_rec))

    - 输出：logits [B, 1, H/4, W/4]
    """

    def __init__(
        self,
        ch2: int,
        ch3: int,
        ch4: int,
        inner_ch: int = 128,
    ):
        super().__init__()

        # 把不同层的通道统一映射到 inner_ch
        self.adapt2 = nn.Conv2d(ch2, inner_ch, kernel_size=1)
        self.adapt3 = nn.Conv2d(ch3, inner_ch, kernel_size=1)
        self.adapt4 = nn.Conv2d(ch4, inner_ch, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(inner_ch)
        self.bn3 = nn.BatchNorm2d(inner_ch)
        self.bn4 = nn.BatchNorm2d(inner_ch)
        self.act = nn.ReLU(inplace=True)

        # 最终融合 head：3 个尺度的特征拼接 → conv → logit
        self.out_conv = nn.Sequential(
            nn.Conv2d(inner_ch * 3, inner_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        f2_orig=None,
        f3_orig=None,
        f4_orig=None,
        f2_rec=None,
        f3_rec=None,
        f4_rec=None,
        feats_orig=None,
        feats_rec=None,
    ):
        """
        支持两种调用方式：
        1) head(f2_orig, f3_orig, f4_orig, f2_rec, f3_rec, f4_rec)
        2) head(feats_orig=(f2_o,f3_o,f4_o), feats_rec=(f2_r,f3_r,f4_r))
        """

        # -------- 兼容关键字调用：feats_orig / feats_rec --------
        if feats_orig is not None and feats_rec is not None:
            # feats_orig 和 feats_rec 都是 (f2, f3, f4)
            f2_orig, f3_orig, f4_orig = feats_orig
            f2_rec, f3_rec, f4_rec = feats_rec

        # 做一点简单检查，避免 silent bug
        assert (
            f2_orig is not None
            and f3_orig is not None
            and f4_orig is not None
            and f2_rec is not None
            and f3_rec is not None
            and f4_rec is not None
        ), "[Stage2FeatureDiffHead] forward() 缺少必要的特征输入。"

        # -------- 多尺度特征差分 --------
        # 绝对值差：越大表示重构不一致 → 疑似异常
        d2 = torch.abs(f2_orig - f2_rec)
        d3 = torch.abs(f3_orig - f3_rec)
        d4 = torch.abs(f4_orig - f4_rec)

        # 通道映射 + BN + ReLU
        d2 = self.act(self.bn2(self.adapt2(d2)))
        d3 = self.act(self.bn3(self.adapt3(d3)))
        d4 = self.act(self.bn4(self.adapt4(d4)))

        # 统一到 f2 的空间尺寸
        h2, w2 = d2.shape[-2], d2.shape[-1]
        d3_up = F.interpolate(d3, size=(h2, w2), mode="bilinear", align_corners=False)
        d4_up = F.interpolate(d4, size=(h2, w2), mode="bilinear", align_corners=False)

        # 拼接三个尺度的差分特征
        feat_cat = torch.cat([d2, d3_up, d4_up], dim=1)  # [B, 3*inner_ch, H/4, W/4]

        # 输出一张 logit anomaly map
        logits = self.out_conv(feat_cat)  # [B, 1, H/4, W/4]
        return logits
