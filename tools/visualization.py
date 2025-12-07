# tools/visualization.py
import os
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt


def save_triplet_heatmaps(
    images_raw: np.ndarray,
    anomaly_maps: np.ndarray,
    names: List[str],
    save_dir: str,
    cmap=cv2.COLORMAP_JET,
):
    """
    images_raw: [N, H, W, 3] uint8, RGB
    anomaly_maps: [N, H, W] float (未归一化 / 已归一化都行)
    names: 每张图的名字
    """
    os.makedirs(save_dir, exist_ok=True)

    # 归一化到 0~1
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    maps_min = maps.min()
    maps = maps - maps_min
    maps_max = maps.max()
    if maps_max > 0:
        maps = maps / maps_max

    for img, amap, name in zip(images_raw, maps, names):
        # 原图：RGB
        img_rgb = img.astype(np.uint8)

        # heatmap 图（只显示热力图）
        amap_255 = (amap * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(amap_255, cmap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # 叠加图
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(img_bgr, 0.4, heatmap_bgr, 0.6, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img_rgb)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(heatmap_rgb)
        axs[1].set_title("Heatmap")
        axs[1].axis("off")

        axs[2].imshow(overlay_rgb)
        axs[2].set_title("Overlay")
        axs[2].axis("off")

        fig.tight_layout()
        out_path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
