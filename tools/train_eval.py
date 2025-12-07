# tools_stage1_combine/train_eval.py

import os
import shutil
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modules_stage1_combine.simplenet import SimpleNet
from .dataset import MVTecDataset
from .metrics import compute_image_auroc, compute_pixel_auroc
from .visualization import save_triplet_heatmaps


def make_dataloaders(cfg) -> Dict[str, Any]:
    train_dataset = MVTecDataset(
        root=cfg.data_root,
        category=cfg.category,
        split="train",
        resize=cfg.image_size,
        crop_size=cfg.image_size,
    )
    test_dataset = MVTecDataset(
        root=cfg.data_root,
        category=cfg.category,
        split="test",
        resize=cfg.image_size,
        crop_size=cfg.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }


def train_one_epoch(
    model: SimpleNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_weight: float = 1.0,
) -> float:
    """
    SimpleNet 的训练策略：
    - 对每张正常图像提取特征
    - 适配后得到 feat_n
    - 加噪得到 feat_a
    - 拼成一个 batch，判别器做二分类 (normal=0, anomaly=1)
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_steps = 0

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)

        with torch.no_grad():
            feat = model.encode(imgs)  # [B, C, H, W]
        feat = model.adapt(feat)      # [B, D, H, W]

        # 正常特征
        feat_n = feat
        # 伪异常特征（特征级 G）
        feat_a = model.generate_anomaly_feat(feat)

        feats_all = torch.cat([feat_n, feat_a], dim=0)  # [2B, D, H, W]
        logits_all = model.discriminate(feats_all)      # [2B, 1, H, W]

        B = feat.size(0)
        H, W = logits_all.shape[-2:]
        labels = torch.cat(
            [
                torch.zeros(B, 1, H, W, device=device),
                torch.ones(B, 1, H, W, device=device) * noise_weight,
            ],
            dim=0,
        )

        loss = criterion(logits_all, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate(
    model: SimpleNet,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    heatmap_dir: str = None,
) -> Dict[str, float]:
    model.eval()

    image_scores = []  # image-level score (max of anomaly map)
    image_labels = []

    all_masks = []
    all_amaps = []
    all_raw_images = []
    all_names = []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy()
        masks = batch["mask"]  # numpy [B,H,W]
        raw_imgs = batch["image_raw"].numpy()  # [B,H,W,3]
        names = batch["name"]

        # anomaly map in feature space
        amap = model(imgs)  # [B,1,h,w]
        amap = F.interpolate(
            amap,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        amap = amap.squeeze(1)  # [B,H,W]
        amap_np = amap.cpu().numpy()

        # image-level score：取 anomaly map 最大值
        scores = amap_np.reshape(amap_np.shape[0], -1).max(axis=1)

        image_scores.extend(scores.tolist())
        image_labels.extend(labels.tolist())

        all_masks.append(masks)
        all_amaps.append(amap_np)
        all_raw_images.append(raw_imgs)
        all_names.extend(list(names))

    all_masks = np.concatenate(all_masks, axis=0)
    all_amaps = np.concatenate(all_amaps, axis=0)
    all_raw_images = np.concatenate(all_raw_images, axis=0)

    img_metrics = compute_image_auroc(image_scores, image_labels)
    pix_metrics = compute_pixel_auroc(all_amaps, all_masks)

    if heatmap_dir is not None:
        save_triplet_heatmaps(
            images_raw=all_raw_images,
            anomaly_maps=all_amaps,
            names=all_names,
            save_dir=heatmap_dir,
        )

    return {
        "image_auroc": float(img_metrics["auroc"]),
        "pixel_auroc": float(pix_metrics["auroc"]),
    }


def run_train(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_txt = os.path.join(cfg.output_dir, "results.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = make_dataloaders(cfg)
    train_loader = dataloaders["train_loader"]
    test_loader = dataloaders["test_loader"]

    # 1) 构建 SimpleNet
    model = SimpleNet(
        backbone_name=cfg.backbone,
        feature_dim=cfg.feature_dim,
        noise_sigma=cfg.noise_sigma,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    # ✅ 2) 如果存在 DNP 预训练的 encoder，就加载进来（G+F 真正接入 SimpleNet）
    pretrained_enc_path = os.path.join(cfg.output_dir, "pretrained_encoder.pth")
    if os.path.isfile(pretrained_enc_path):
        print(f"[SimpleNet] Loading pretrained encoder from {pretrained_enc_path}")
        enc_state = torch.load(pretrained_enc_path, map_location=device)
        model.encoder.load_state_dict(enc_state, strict=True)
        print("[SimpleNet] Encoder weights loaded from DNP (G+F).")
    else:
        print("[SimpleNet] No pretrained_encoder.pth found, use ImageNet backbone.")

    # 3) 再建优化器（一定要在 load 之后再建）
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_score = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(cfg.output_dir, "best_simplenet.pth")
    best_heatmap_dir = os.path.join(cfg.output_dir, "best_heatmaps")

    with open(log_txt, "w", encoding="utf-8") as f:
        f.write("# epoch\ttrain_loss\timage_auroc\tpixel_auroc\tmean_auroc\n")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            noise_weight=cfg.noise_weight,
        )

        # 评估
        tmp_heatmap_dir = os.path.join(cfg.output_dir, f"temp_heatmaps_epoch_{epoch}")
        if os.path.exists(tmp_heatmap_dir):
            shutil.rmtree(tmp_heatmap_dir)

        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            image_size=cfg.image_size,
            heatmap_dir=tmp_heatmap_dir,
        )

        img_auc = metrics["image_auroc"]
        pix_auc = metrics["pixel_auroc"]
        mean_auc = (img_auc + pix_auc) / 2.0

        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch}\t{train_loss:.6f}\t{img_auc:.4f}\t{pix_auc:.4f}\t{mean_auc:.4f}\n"
            )

        print(
            f"[Epoch {epoch:03d}] loss={train_loss:.4f}, "
            f"image_auroc={img_auc:.4f}, pixel_auroc={pix_auc:.4f}, mean={mean_auc:.4f}"
        )

        # 只保留综合最高的权重 & 对应热力图
        if mean_auc > best_score:
            best_score = mean_auc
            best_epoch = epoch

            # 保存权重
            torch.save(model.state_dict(), best_ckpt_path)

            # 删掉旧热力图目录，保留当前最好的一份
            if os.path.exists(best_heatmap_dir):
                shutil.rmtree(best_heatmap_dir)
            shutil.move(tmp_heatmap_dir, best_heatmap_dir)
            print(f"==> New best at epoch {epoch}, mean_auroc={mean_auc:.4f}")
        else:
            # 不是 best，删掉临时热力图
            if os.path.exists(tmp_heatmap_dir):
                shutil.rmtree(tmp_heatmap_dir)

    print(f"Training finished. Best epoch = {best_epoch}, best mean_auroc = {best_score:.4f}")
    print(f"Best weights saved to: {best_ckpt_path}")
    print(f"Best heatmaps saved to: {best_heatmap_dir}")
