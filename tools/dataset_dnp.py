# tools/dataset_dnp.py
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .anomaly_generator import PerlinAnomalyGenerator


class DNPDataset(Dataset):
    """
    Stage I (DNP) 用的数据集：
    - 只使用 MVTec 的 train/good 正常图
    - 对每张图用 G 生成伪异常 (IG, mask)
    """

    def __init__(
        self,
        data_root: str,
        category: str,
        texture_root: str,
        image_size: int = 256,
    ):
        self.data_root = Path(data_root)
        self.category = category

        img_dir = self.data_root / category / "train" / "good"
        self.img_paths: List[Path] = sorted(img_dir.glob("*.png"))
        if not self.img_paths:
            raise RuntimeError(f"No train/good images at {img_dir}")

        self.generator = PerlinAnomalyGenerator(texture_root=texture_root)

        self.img_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=Image.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=Image.NEAREST),
                transforms.CenterCrop(image_size),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Dict[str, Any]:
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")

        # 伪异常图 IG + mask（在原始分辨率上生成）
        img_ano, mask_np = self.generator(img)  # img_ano: PIL, mask_np: [H,W] 0/1

        img_ano_t = self.img_transform(img_ano)

        mask_img = Image.fromarray(mask_np * 255)
        mask_resized = self.mask_transform(mask_img)
        mask_resized = np.array(mask_resized, dtype=np.uint8)
        mask_resized = (mask_resized > 0).astype(np.uint8)

        return {
            "image_ano": img_ano_t,    # 输入给 F+D 的伪异常图
            "mask": mask_resized,      # 0/1, H×W
            "name": path.stem,
        }
