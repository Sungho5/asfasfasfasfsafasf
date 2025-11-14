"""
DSRN Configuration
"""
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path


@dataclass
class DSRNConfig:
    # ===== Data =====
    data_root: str = "/home/imgadmin/DATA1/sungho/synology/sungho/NEW_Anomaly/data/2048/"
    train_split: float = 0.9
    train_ratio: float = 0.9  # For dataset split
    image_size: int = 256

    # DICOM window level normalization
    window_center: int = 2048
    window_width: int = 4096

    # ===== Model Architecture =====
    base_channels: int = 64
    feature_dims: Tuple[int, int, int, int] = (64, 128, 256, 512)
    num_prototypes: int = 1000
    feature_dim: int = 512  # For anomaly scorer

    # Anomaly scorer
    scorer_channels: Tuple[int, int, int] = (256, 128, 1)

    # Normal stream
    normal_alpha: float = 0.05  # Very small residual

    # Abnormal stream
    context_channels: int = 64
    decoder_channels: Tuple[int, int] = (512, 256)

    # Texture synthesizer
    texture_attention_dim: int = 256

    # ===== Training =====
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Loss weights
    lambda_recon: float = 1.0
    lambda_anomaly: float = 0.5
    lambda_identity: float = 0.3
    lambda_perceptual: float = 0.2

    # ===== Synthetic Lesion =====
    num_lesions_range: Tuple[int, int] = (2, 4)
    lesion_type: str = 'random'  # 'random', 'radiolucent', 'mixed'
    lesion_prob: float = 0.8  # Probability of adding lesion
    min_lesion_radius: int = 10
    max_lesion_radius: int = 40

    # ===== Prototype Learning =====
    prototype_update_freq: int = 5  # epochs

    # ===== Hardware =====
    device: str = "cuda"
    num_workers: int = 4

    # ===== Paths =====
    save_dir: str = "./outputs/dsrn"
    checkpoint_dir: str = "./outputs/dsrn/checkpoints"
    log_dir: str = "./outputs/dsrn/logs"
    output_dir: str = "./outputs/dsrn/outputs"

    # ===== Checkpointing =====
    checkpoint_freq: int = 5

    # ===== Visualization =====
    vis_freq: int = 1
    num_vis_samples: int = 4
    visualization_samples: int = 10

    def __post_init__(self):
        """Create directories after initialization"""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = DSRNConfig()
