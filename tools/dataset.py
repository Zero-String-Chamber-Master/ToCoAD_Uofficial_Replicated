# tools/dataset.py
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    """
    适用于 MVTec AD 的数据集封装。
    目录结构遵循官方：
        root/category/train/good/xxx.png
        root/category/test/good/xxx.png
        root/category/test/defect_type/xxx.png
        root/category/ground_truth/defect_type/xxx_mask.png
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        resize: int = 256,
        crop_size: int = 256,
    ):
        assert split in ["train", "test"]
        self.root = Path(root)
        self.category = category
        self.split = split

        self.resize = resize
        self.crop_size = crop_size

        self.img_paths = []
        self.mask_paths = []
        self.labels = []  # 0 normal, 1 defect
        self.names = []

        if split == "train":
            img_dir = self.root / category / "train" / "good"
            for p in sorted(img_dir.glob("*.png")):
                self.img_paths.append(p)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.names.append(p.stem)
        else:
            test_dir = self.root / category / "test"
            gt_dir = self.root / category / "ground_truth"
            for defect_type_dir in sorted(test_dir.iterdir()):
                if not defect_type_dir.is_dir():
                    continue
                defect_type = defect_type_dir.name  # "good" or defect class
                for p in sorted(defect_type_dir.glob("*.png")):
                    self.img_paths.append(p)
                    if defect_type == "good":
                        self.mask_paths.append(None)
                        self.labels.append(0)
                    else:
                        # mask 文件名：xxx_mask.png
                        mask_path = gt_dir / defect_type / f"{p.stem}_mask.png"
                        self.mask_paths.append(mask_path)
                        self.labels.append(1)
                    self.names.append(f"{defect_type}_{p.stem}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resize, interpolation=Image.BILINEAR),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.raw_transform = transforms.Compose(
            [
                transforms.Resize(self.resize, interpolation=Image.BILINEAR),
                transforms.CenterCrop(self.crop_size),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        name = self.names[idx]

        img = Image.open(img_path).convert("RGB")
        img_raw = self.raw_transform(img)  # 用于可视化（未归一化）

        if mask_path is not None:
            mask_img = Image.open(mask_path).convert("L")
            mask_img = self.raw_transform(mask_img)
            mask = np.array(mask_img, dtype=np.uint8)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)

        img_tensor = self.transform(img)

        return {
            "image": img_tensor,
            "image_raw": np.array(img_raw, dtype=np.uint8),
            "label": int(label),
            "mask": mask,
            "name": name,
        }
