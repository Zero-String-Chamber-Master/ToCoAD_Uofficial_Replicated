# tools_stage1_combine/train_stage1_dnp.py

import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from modules_stage1_combine.feature_extractor import ResNetFeatureExtractor
from modules_stage1_combine.discriminative_unet import DiscriminativeUNet
from .dataset_dnp import DNPDataset
from .focal_loss import FocalLoss, DiceLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_stage1_dnp(cfg: Any):
    """
    Stage I: Discriminative Network Pretraining (DNP)
    - 用 G 生成的伪异常图 IG 和 mask
    - 训练 decoder D 做像素级二分类
    - F(encoder) 可选是否微调 (cfg.dnp_freeze_backbone)
    """
    # 固定随机种子，保证多次实验可比较
    set_seed(1234)

    os.makedirs(cfg.dnp_output_dir, exist_ok=True)
    log_txt = os.path.join(cfg.dnp_output_dir, "dnp_results.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DNPDataset(
        data_root=cfg.data_root,
        category=cfg.category,
        texture_root=cfg.texture_root,
        image_size=cfg.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.dnp_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Backbone F
    encoder = ResNetFeatureExtractor(
        name=cfg.backbone,
        pretrained=True,
        frozen=cfg.dnp_freeze_backbone,  # True: 只训练 D；False: F+D 一起训
    ).to(device)

    # 需要知道各层通道数
    with torch.no_grad():
        sample = next(iter(loader))
        img_ano = sample["image_ano"].to(device)
        f2, f3, f4 = encoder(img_ano)
        ch2, ch3, ch4 = f2.shape[1], f3.shape[1], f4.shape[1]

    decoder = DiscriminativeUNet(
        ch2=ch2,
        ch3=ch3,
        ch4=ch4,
        base_ch=cfg.dnp_base_channels,
    ).to(device)

    # 优化器：encoder（可选）+ decoder
    params = []
    if not cfg.dnp_freeze_backbone:
        params.append({"params": encoder.parameters(), "lr": cfg.dnp_lr * 0.1})
    params.append({"params": decoder.parameters(), "lr": cfg.dnp_lr})

    optimizer = torch.optim.Adam(
        params,
        weight_decay=cfg.dnp_weight_decay,
    )

    # loss: Focal + lambda_dice * Dice
    criterion_focal = FocalLoss(alpha=cfg.dnp_focal_alpha,
                                gamma=cfg.dnp_focal_gamma)
    criterion_dice = DiceLoss(smooth=1.0)
    lambda_dice = getattr(cfg, "dnp_lambda_dice", 0.5)  # 没配就默认 0.5

    best_loss = float("inf")
    best_ckpt_path = os.path.join(cfg.dnp_output_dir, "dnp_best.pth")

    with open(log_txt, "w", encoding="utf-8") as f:
        f.write("# epoch\tloss_focal\tloss_dice\ttotal_loss\n")

    for epoch in range(1, cfg.dnp_epochs + 1):
        encoder.train(not cfg.dnp_freeze_backbone)
        decoder.train()

        total_loss = 0.0
        total_focal = 0.0
        total_dice = 0.0
        n_steps = 0

        for batch in loader:
            img_ano = batch["image_ano"].to(device, non_blocking=True)  # [B,3,H,W]
            mask_np = batch["mask"]  # 可能是 numpy 或 Tensor，[B,H,W]

            # 🔧 这里做兼容：既支持 numpy，又支持 tensor
            if isinstance(mask_np, torch.Tensor):
                mask = mask_np.to(device=device, dtype=torch.long)  # [B,H,W]
            else:
                mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.long)

            # 前向：F + D
            f2, f3, f4 = encoder(img_ano)
            logits = decoder((f2, f3, f4))  # [B,2,h,w]

            # FocalLoss / DiceLoss 内部会自己把 target resize 到 logits 尺寸
            loss_f = criterion_focal(logits, mask)  # mask: [B,H,W] 0/1
            loss_d = criterion_dice(logits, mask)
            loss = loss_f + lambda_dice * loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_focal += loss_f.item()
            total_dice += loss_d.item()
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        avg_f = total_focal / max(n_steps, 1)
        avg_d = total_dice / max(n_steps, 1)

        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{avg_f:.6f}\t{avg_d:.6f}\t{avg_loss:.6f}\n")

        print(
            f"[DNP Epoch {epoch:03d}] "
            f"loss_focal={avg_f:.4f}, loss_dice={avg_d:.4f}, total={avg_loss:.4f}"
        )

        # 只保留最小 total loss 的 ckpt
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                },
                best_ckpt_path,
            )
            print(f"==> New best DNP at epoch {epoch}, loss={avg_loss:.4f}")

    print(f"[DNP] Training finished. Best loss = {best_loss:.4f}")
    print(f"[DNP] Best checkpoint saved to: {best_ckpt_path}")
