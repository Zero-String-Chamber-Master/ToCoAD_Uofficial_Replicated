# main_full_tocoad.py
"""
一键管三件事：
1) （可选）Stage1: DNP 预训练 (G+F+D)
2) （可选）Stage2: 重构 + head (F2+R+H)
3) Stage3: SimpleNet 检测训练 + 评估
   - SimpleNet 里用 DNP encoder (G+F)
   - evaluate 里融合 DNP decoder + Stage2 重构分支
"""

from dataclasses import dataclass

from tools.train_stage1_dnp import run_stage1_dnp
from tools.train_eval import run_train
from tools.train_stage2 import Stage2Config, run_stage2_reconstruct


@dataclass
class Config:
    # ========= 通用数据 & backbone =========
    data_root: str = "datasets"
    category: str = "cable"
    texture_root: str = "dtd/images"
    image_size: int = 256
    num_workers: int = 4
    backbone: str = "wide_resnet50_2"

    # ===== Memory Bank 设置 =====
    use_memory_bank: bool = True          # 开 / 关 memory 分支
    memory_max_samples: int = 50000       # memory 中最多保存多少个 patch
    memory_beta: float = 0.3              # memory map 在 pixel 融合中的权重

    # ========= Stage1: DNP =========
    use_dnp_stage1: bool = True
    dnp_output_dir: str = "./cable_stageA_final_dnp"
    dnp_batch_size: int = 4
    dnp_epochs: int = 50
    dnp_lr: float = 1e-4
    dnp_weight_decay: float = 1e-5
    dnp_base_channels: int = 256
    dnp_focal_alpha: float = 0.25
    dnp_focal_gamma: float = 2.0
    dnp_freeze_backbone: bool = True  # True: 只训 D；False: F+D 一起训

    # ========= Stage2: 重构 + head =========
    use_stage2_train: bool = True
    stage2_output_dir: str = "./cable_stageA_final_discf"
    stage2_batch_size: int = 4
    stage2_epochs: int = 10          # 建议先小一点
    stage2_lr: float = 1e-4
    stage2_weight_decay: float = 1e-5
    stage2_base_channels: int = 256
    stage2_freeze_backbone: bool = True
    stage2_fg_weight: float = 2.0
    stage2_bg_weight: float = 1.0
    # head & 损失权重
    stage2_inner_ch: int = 128
    stage2_lambda_recon: float = 1.0
    stage2_lambda_head: float = 1.0

    # ========= Stage3: SimpleNet =========
    feature_dim: int = 256
    noise_sigma: float = 0.5
    freeze_backbone: bool = True
    batch_size: int = 8
    test_batch_size: int = 8
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    noise_weight: float = 1.0
    output_dir: str = "./cable_stageA_final"

    # ========= 融合相关（在 train_eval.evaluate 里用） =========
    use_stage2_fusion: bool = True   # evaluate 里面用来决定要不要加 Stage2
    stage2_gamma: float = 0.3        # SimpleNet/DNP vs Stage2 的融合权重


def main():
    cfg = Config()

    # ---------------- Stage1: DNP ----------------
    if cfg.use_dnp_stage1 and cfg.dnp_epochs > 0:
        print("========== Stage1: DNP pretraining ==========")
        run_stage1_dnp(cfg)
    else:
        print("Skip Stage1 DNP.")

    # ---------------- Stage2: 重构 + head ----------------
    if cfg.use_stage2_train and cfg.stage2_epochs > 0:
        print("========== Stage2: Reconstruction + Head ==========")
        # 注意：这里把 cfg 的字段拆成 Stage2Config
        s2_cfg = Stage2Config(
            data_root=cfg.data_root,
            category=cfg.category,
            texture_root=cfg.texture_root,
            image_size=cfg.image_size,
            backbone=cfg.backbone,
            dnp_output_dir=cfg.dnp_output_dir,
            stage2_init_from_dnp=True,               # 用 Stage1 的 encoder 初始化
            stage2_output_dir=cfg.stage2_output_dir,
            stage2_batch_size=cfg.stage2_batch_size,
            stage2_epochs=cfg.stage2_epochs,
            stage2_lr=cfg.stage2_lr,
            stage2_weight_decay=cfg.stage2_weight_decay,
            stage2_base_channels=cfg.stage2_base_channels,
            stage2_freeze_backbone=cfg.stage2_freeze_backbone,
            stage2_fg_weight=cfg.stage2_fg_weight,
            stage2_bg_weight=cfg.stage2_bg_weight,
        )
        # 把新增的 3 个超参数挂上去（train_stage2 里用 getattr 取）
        s2_cfg.stage2_inner_ch = cfg.stage2_inner_ch
        s2_cfg.stage2_lambda_recon = cfg.stage2_lambda_recon
        s2_cfg.stage2_lambda_head = cfg.stage2_lambda_head

        run_stage2_reconstruct(s2_cfg)
    else:
        print("Skip Stage2 training.")

    # ---------------- Stage3: SimpleNet + 融合 ----------------
    print("========== Stage3: SimpleNet training + evaluation ==========")
    # 这里直接把 cfg 传给你当前的 tools_stage1_5.train_eval.run_train
    # 要求：train_eval.py 里已经支持：
    #   - 从 cfg.dnp_output_dir 读取 DNP
    #   - 从 cfg.stage2_output_dir 读取 Stage2 模型
    #   - 通过 cfg.use_stage2_fusion 和 cfg.stage2_gamma 控制是否融合
    run_train(cfg)


if __name__ == "__main__":
    main()
