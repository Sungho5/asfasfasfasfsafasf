"""
Configuration for DSRN (Dual-Stream Selective Reconstruction Network)
"""

import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DSRNConfig:
    """DSRN Configuration"""

    # ===== Data =====
    data_root: str = "/dataset/panorama_total"
    image_size: int = 256
    train_ratio: float = 0.8

    # ===== Model =====
    in_channels: int = 1
    base_channels: int = 64
    num_prototypes: int = 1000
    feature_dim: int = 512

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

    # ===== Lesion Synthesis =====
    lesion_prob: float = 0.8  # Probability of adding lesion
    min_lesion_radius: int = 10
    max_lesion_radius: int = 40

    # ===== Optimization =====
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Paths =====
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"

    # ===== Testing =====
    visualization_samples: int = 10

    def __post_init__(self):
        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# Default configuration
default_config = DSRNConfig()
