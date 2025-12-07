# tools/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    多分类 Focal loss，适合处理极度不平衡的前景/背景。
    用在 ToCoAD/DNP 的像素级二分类上：
    - num_classes = 2
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """
        logits: [B,C,H,W]
        target: [B,H,W]  取值 0~C-1
        """
        b, c, h, w = logits.shape
        if target.shape[-2:] != (h, w):
            # 保证尺寸一致
            target = F.interpolate(target.unsqueeze(1).float(),
                                   size=(h, w),
                                   mode="nearest").long().squeeze(1)

        log_prob = F.log_softmax(logits, dim=1)  # [B,C,H,W]
        prob = log_prob.exp()

        # one-hot
        target_oh = F.one_hot(target, num_classes=c)  # [B,H,W,C]
        target_oh = target_oh.permute(0, 3, 1, 2).float()  # [B,C,H,W]

        # 只取真类的 prob/log_prob
        prob_t = (prob * target_oh).sum(dim=1)       # [B,H,W]
        log_prob_t = (log_prob * target_oh).sum(dim=1)

        # Focal 部分
        focal_weight = (1 - prob_t) ** self.gamma

        # 类别权重：假设 1 是前景，0 是背景
        alpha_map = self.alpha * target_oh[:, 1] + (1 - self.alpha) * target_oh[:, 0]

        loss = -alpha_map * focal_weight * log_prob_t  # [B,H,W]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    二分类 Dice loss：
    - logits: [B,2,H,W]
    - target: [B,H,W]，0/1
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        b, c, h, w = logits.shape

        # 1) 先把 target 的空间尺寸对齐到 logits
        if target.shape[-2:] != (h, w):
            target = F.interpolate(
                target.unsqueeze(1).float(),  # [B,1,H_orig,W_orig]
                size=(h, w),
                mode="nearest",
            ).squeeze(1)                       # [B,H,W]
        else:
            target = target.float()            # [B,H,W]

        # 2) 取异常类的概率
        probs = torch.softmax(logits, dim=1)[:, 1]  # [B,H,W]

        # 3) 展平后计算 Dice
        probs = probs.view(b, -1)
        target = target.view(b, -1)

        intersection = (probs * target).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        loss = 1.0 - dice  # [B]
        return loss.mean()

