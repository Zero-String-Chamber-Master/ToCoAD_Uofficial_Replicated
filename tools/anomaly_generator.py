# tools/anomaly_generator.py
import os
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
import cv2


def _load_texture_paths(texture_root: str) -> List[Path]:
    root = Path(texture_root)
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(p)
    if not paths:
        raise RuntimeError(f"No texture images found in {texture_root}")
    return paths


def _random_rotate90(img: np.ndarray) -> np.ndarray:
    """随机旋转 0/90/180/270 度."""
    k = random.randint(0, 3)
    return np.rot90(img, k, axes=(0, 1))


def _generate_smooth_noise(h: int, w: int,
                           min_grid: int = 8,
                           max_grid: int = 32,
                           octaves: int = 2) -> np.ndarray:
    """
    生成类似 Perlin 的平滑噪声：
    - 先在小分辨率上随机
    - 再用插值 + 高斯模糊放大
    - 多个频率叠加
    """
    noise = np.zeros((h, w), dtype=np.float32)
    for _ in range(octaves):
        gh = random.randint(min_grid, max_grid)
        gw = max(gh * w // h, 4)
        small = np.random.rand(gh, gw).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (5, 5), 0)
        noise += up
    noise = noise - noise.min()
    if noise.max() > 0:
        noise = noise / noise.max()
    return noise


class PerlinAnomalyGenerator:
    """
    图像级伪异常生成器 G：
    输入：一张正常图（PIL.Image, RGB）
    输出：伪异常图 + 二值 mask
    """

    def __init__(
        self,
        texture_root: str,
        min_coverage: float = 0.01,
        max_coverage: float = 0.2,
        perlin_min_grid: int = 8,
        perlin_max_grid: int = 32,
        perlin_octaves: int = 2,
    ):
        """
        texture_root: 纹理库路径（可以放 DTD / 其它纹理图片）
        min_coverage, max_coverage: 缺陷区域占图像比例的范围
        """
        self.texture_paths = _load_texture_paths(texture_root)
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.perlin_min_grid = perlin_min_grid
        self.perlin_max_grid = perlin_max_grid
        self.perlin_octaves = perlin_octaves

    def _random_texture_patch(self, h: int, w: int) -> np.ndarray:
        """随机选择一张纹理图，中心裁剪到目标大小，并做随机旋转."""
        path = random.choice(self.texture_paths)
        tex = Image.open(path).convert("RGB")
        th, tw = tex.size[1], tex.size[0]

        # 先按照较短边缩放，让纹理覆盖完整区域
        scale = max(h / th, w / tw)
        new_tw, new_th = int(tw * scale + 0.5), int(th * scale + 0.5)
        tex = tex.resize((new_tw, new_th), Image.BILINEAR)

        # 中心裁剪到 (w, h)
        left = (new_tw - w) // 2
        top = (new_th - h) // 2
        tex = tex.crop((left, top, left + w, top + h))

        tex_np = np.array(tex, dtype=np.uint8)
        tex_np = _random_rotate90(tex_np)
        return tex_np

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """
        img: PIL.Image (RGB)
        return:
            img_ano: PIL.Image (RGB) 带缺陷图
            mask: np.uint8[H,W] 0/1 缺陷区域
        """
        img = img.convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        h, w = img_np.shape[0], img_np.shape[1]

        # ---------- 1) 生成平滑噪声 & 更稳的覆盖率控制 ----------
        noise = _generate_smooth_noise(
            h, w,
            min_grid=self.perlin_min_grid,
            max_grid=self.perlin_max_grid,
            octaves=self.perlin_octaves,
        )

        target_cov = 0.5 * (self.min_coverage + self.max_coverage)
        best_mask = None
        best_diff = 1e9

        for _ in range(10):
            thr = random.uniform(0.4, 0.8)
            m = (noise > thr).astype(np.uint8)
            cov = m.mean()
            diff = abs(cov - target_cov)
            if diff < best_diff:
                best_diff = diff
                best_mask = m
            if self.min_coverage <= cov <= self.max_coverage:
                best_mask = m
                break

        mask = best_mask.astype(np.uint8)

        # 形态学平滑一下 mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ---------- 2) 计算 soft mask（边缘羽化） ----------
        inv_mask = 1 - mask
        dist = cv2.distanceTransform(inv_mask.astype(np.uint8), cv2.DIST_L2, 5)
        if dist.max() > 0:
            dist = dist / dist.max()
        soft_mask = (1.0 - dist) * mask  # 中心区域 ~1，外圈 ~0，mask 外仍为 0
        soft_mask_3 = np.stack([soft_mask] * 3, axis=-1).astype(np.float32)

        # ---------- 3) 随机选择缺陷类型：纹理 or 亮度 ----------
        defect_type = np.random.choice(["texture", "brightness"], p=[0.7, 0.3])

        img_f = img_np.astype(np.float32) / 255.0

        if defect_type == "texture":
            tex_np = self._random_texture_patch(h, w)
            tex_f = tex_np.astype(np.float32) / 255.0

            # 随机融合强度
            alpha = np.random.uniform(0.5, 1.0)

            blend = img_f * (1.0 - alpha * soft_mask_3) + tex_f * (alpha * soft_mask_3)
            img_ano_np = (blend * 255.0).clip(0, 255).astype(np.uint8)

        else:
            # brightness 型缺陷：局部变亮/变暗
            delta = np.random.uniform(-0.5, 0.5)
            img_ano_f = img_f + delta * soft_mask_3
            img_ano_f = np.clip(img_ano_f, 0.0, 1.0)
            img_ano_np = (img_ano_f * 255.0).astype(np.uint8)

        # ---------- 4) 最后整体随机旋转 ----------
        img_ano_np = _random_rotate90(img_ano_np)
        mask = _random_rotate90(mask)

        img_ano = Image.fromarray(img_ano_np)
        return img_ano, mask.astype(np.uint8)
