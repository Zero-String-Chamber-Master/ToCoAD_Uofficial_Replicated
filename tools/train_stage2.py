# tools_stage2_tocoad/train_stage2.py
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ===== 按你的工程路径改这里 =====
from modules.feature_extractor import ResNetFeatureExtractor  # ← 特征提取 F2
from modules.reconstructor import ReconstructionUNet    # ← 重构 R
from tools.anomaly_generator import PerlinAnomalyGenerator   # ← G：伪异常生成器


# ----------------- Dataset -----------------
class Stage2ReconstructDataset(Dataset):
    """
    Stage2 用的数据集：
    - 使用 MVTec train/good 正常图
    - 用 G 生成 (x_ano, mask)
    - 目标是从 x_ano 重构回 x_clean
    """

    def __init__(
        self,
        data_root: str,
        category: str,
        texture_root: str,
        image_size: int = 256,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.category = category

        img_dir = self.data_root / category / "train" / "good"
        exts = [".png", ".jpg", ".jpeg", ".bmp"]
        self.img_paths: List[Path] = [
            p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts
        ]
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No train/good images found in {img_dir}")

        # 图像增强（和 Stage1 / SimpleNet 保持一致即可）
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        # mask 也要 resize 到同样大小
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            ]
        )

        # 伪异常生成器 G
        self.generator = PerlinAnomalyGenerator(texture_root=texture_root)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")

        # 干净图
        img_clean_t = self.img_transform(img)  # [3,H,W]

        # 伪异常图 + mask
        img_ano, mask_np = self.generator(img)  # img_ano: PIL, mask_np: [H0,W0] 0/1
        img_ano_t = self.img_transform(img_ano)

        # mask resize 到网络输入尺寸
        mask_img = Image.fromarray(mask_np.astype(np.uint8) * 255)
        mask_resized = self.mask_transform(mask_img)
        mask_resized = np.array(mask_resized, dtype=np.uint8)
        mask_resized = (mask_resized > 0).astype(np.uint8)  # [H,W] 0/1

        return {
            "image_clean": img_clean_t,
            "image_ano": img_ano_t,
            "mask": mask_resized,
            "name": path.stem,
        }


# ----------------- Utils -----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------- Config（仅给 Stage2 用，如果你需要可以和主 Config 合并） -----------------
@dataclass
class Stage2Config:
    data_root: str
    category: str
    texture_root: str
    image_size: int

    backbone: str

    dnp_output_dir: str
    stage2_init_from_dnp: bool

    stage2_output_dir: str
    stage2_batch_size: int
    stage2_epochs: int
    stage2_lr: float
    stage2_weight_decay: float
    stage2_base_channels: int
    stage2_freeze_backbone: bool

    stage2_fg_weight: float
    stage2_bg_weight: float


# ----------------- 主训练函数 -----------------
def run_stage2_reconstruct(cfg: Stage2Config):
    """
    训练 Stage2 重构模块：
      - F2: ResNetFeatureExtractor
      - R : ReconstructionUNet
    使用带伪异常的 x_ano 作为输入，目标是重构回 x_clean：
      loss = |x_rec - x_clean| 按 mask 进行前景/背景加权
    """
    os.makedirs(cfg.stage2_output_dir, exist_ok=True)
    log_txt = os.path.join(cfg.stage2_output_dir, "stage2_results.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # -------- Dataset & Dataloader --------
    dataset = Stage2ReconstructDataset(
        data_root=cfg.data_root,
        category=cfg.category,
        texture_root=cfg.texture_root,
        image_size=cfg.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.stage2_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # -------- 构建 encoder F2 --------
    encoder = ResNetFeatureExtractor(
        name=cfg.backbone,
        pretrained=True,   # 先用 ImageNet 初始化
        frozen=False,
    ).to(device)

    # 如果需要从 Stage1 DNP 初始化 encoder
    if cfg.stage2_init_from_dnp:
        dnp_ckpt = os.path.join(cfg.dnp_output_dir, "dnp_best.pth")
        if os.path.isfile(dnp_ckpt):
            print(f"[Stage2] Init encoder from DNP: {dnp_ckpt}")
            dnp_state = torch.load(dnp_ckpt, map_location="cpu")
            if "encoder" in dnp_state:
                encoder.load_state_dict(dnp_state["encoder"], strict=False)
            else:
                print("[Stage2] Warning: 'encoder' not found in dnp_best.pth, skip init.")
        else:
            print(f"[Stage2] DNP ckpt not found at {dnp_ckpt}, skip init.")

    # 探测多层通道数
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size, device=device)
        f2, f3, f4 = encoder(dummy)
        ch2, ch3, ch4 = f2.shape[1], f3.shape[1], f4.shape[1]

    # -------- 构建重构网络 R --------
    reconstructor = ReconstructionUNet(
        ch2=ch2,
        ch3=ch3,
        ch4=ch4,
        base_ch=cfg.stage2_base_channels,
        out_channels=3,
    ).to(device)

    # -------- 优化器 --------
    params = []
    if not cfg.stage2_freeze_backbone:
        params.append({"params": encoder.parameters(), "lr": cfg.stage2_lr * 0.1})
    params.append({"params": reconstructor.parameters(), "lr": cfg.stage2_lr})

    optimizer = torch.optim.Adam(
        params,
        weight_decay=cfg.stage2_weight_decay,
    )

    # L1 按像素加权
    l1 = nn.L1Loss(reduction="none")

    best_loss = float("inf")
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write("# epoch\ttrain_loss\n")

    for epoch in range(1, cfg.stage2_epochs + 1):
        encoder.train(not cfg.stage2_freeze_backbone)
        reconstructor.train()

        running_loss = 0.0
        n_steps = 0

        for batch in loader:
            img_clean = batch["image_clean"].to(device)  # [B,3,H,W]
            img_ano = batch["image_ano"].to(device)      # [B,3,H,W]
            mask_np = batch["mask"]  # 可能是 np.ndarray，也可能是 torch.Tensor
            
            # ✅ 关键改动：根据类型分别处理
            if isinstance(mask_np, torch.Tensor):
                # 例如 DataLoader 已经帮你堆成 [B,H,W] Tensor
                mask = mask_np.to(device=device, dtype=torch.float32)
            else:
                # 例如仍然是 numpy 数组 [B,H,W] 或 [H,W]
                mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
        
            # 确保形状是 [B,1,H,W]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

            # 编码伪异常图
            f2, f3, f4 = encoder(img_ano)
            x_rec = reconstructor((f2, f3, f4))  # [B,3,H/4,W/4]
            x_rec_up = torch.nn.functional.interpolate(
                x_rec,
                size=img_clean.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # L1 diff
            diff = l1(x_rec_up, img_clean)  # [B,3,H,W]

            # 前景 / 背景加权
            weight = cfg.stage2_fg_weight * mask + cfg.stage2_bg_weight * (1.0 - mask)
            weight = weight.expand_as(diff)  # [B,3,H,W]

            loss = (diff * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_steps += 1

        epoch_loss = running_loss / max(n_steps, 1)

        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{epoch_loss:.6f}\n")

        print(f"[Stage2][Epoch {epoch:03d}] loss={epoch_loss:.4f}")

        # 保存最好的 ckpt
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = os.path.join(cfg.stage2_output_dir, "stage2_best.pth")
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "reconstructor": reconstructor.state_dict(),
                },
                ckpt_path,
            )
            print(f"==> New best Stage2 at epoch {epoch}, loss={epoch_loss:.4f}")
            print(f"[Stage2] Best weights saved to {ckpt_path}")

    print(f"[Stage2] Training finished. Best loss={best_loss:.4f}")


# ----------------- 推理阶段加载辅助函数（给 SimpleNet / 主框架用） -----------------
def load_stage2_models(cfg, device: torch.device):
    """
    在评估时调用：
        encoder_s2, reconstructor_s2 = load_stage2_models(cfg, device)
    如果找不到 ckpt，则返回 (None, None)。
    """
    ckpt_path = os.path.join(cfg.stage2_output_dir, "stage2_best.pth")
    if not os.path.isfile(ckpt_path):
        print(f"[Stage2] ckpt not found at {ckpt_path}, skip Stage2 fusion.")
        return None, None

    print(f"[Stage2] Loading Stage2 models from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_enc = ckpt.get("encoder", None)
    state_rec = ckpt.get("reconstructor", None)
    if state_enc is None or state_rec is None:
        print("[Stage2] 'encoder' or 'reconstructor' missing in stage2_best.pth.")
        return None, None

    # 构建 encoder
    encoder = ResNetFeatureExtractor(
        name=cfg.backbone,
        pretrained=False,
        frozen=True,
    ).to(device)
    encoder.load_state_dict(state_enc, strict=False)

    # 探测通道
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size, device=device)
        f2, f3, f4 = encoder(dummy)
        ch2, ch3, ch4 = f2.shape[1], f3.shape[1], f4.shape[1]

    # 构建 reconstructor
    reconstructor = ReconstructionUNet(
        ch2=ch2,
        ch3=ch3,
        ch4=ch4,
        base_ch=cfg.stage2_base_channels,
        out_channels=3,
    ).to(device)
    reconstructor.load_state_dict(state_rec, strict=False)

    encoder.eval()
    reconstructor.eval()
    print("[Stage2] Encoder + Reconstructor loaded for inference.")
    return encoder, reconstructor
