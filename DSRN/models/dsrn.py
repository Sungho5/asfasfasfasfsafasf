"""
DSRN - Dual-Stream Selective Reconstruction Network
"""

import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .anomaly_scorer import SpatialAnomalyScorer
from .normal_stream import NormalStream
from .abnormal_stream import AbnormalStream
from .fusion import SoftFusion


class DSRN(nn.Module):
    """
    Dual-Stream Selective Reconstruction Network

    Two independent streams:
    - Normal Stream: Identity pathway for normal regions
    - Abnormal Stream: Perfect reconstruction for abnormal regions

    Soft selection mechanism automatically blends outputs based on anomaly scores
    """

    def __init__(self, config):
        super().__init__()

        # Components
        self.feature_extractor = FeatureExtractor(
            in_channels=config.input_channels,  # Now supports multi-channel input
            base_channels=config.base_channels
        )

        self.anomaly_scorer = SpatialAnomalyScorer(
            num_prototypes=config.num_prototypes,
            feature_dim=config.feature_dim
        )

        self.normal_stream = NormalStream(
            in_channels=config.input_channels,
            alpha=config.normal_alpha
        )

        self.abnormal_stream = AbnormalStream(
            in_channels=config.input_channels
        )

        self.fusion = SoftFusion()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 70)
        print("[DSRN] Dual-Stream Selective Reconstruction Network")
        print("=" * 70)
        print(f"Input channels:        {config.input_channels} [original, CLAHE, gradient]")
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")
        print("=" * 70)

    def forward(self, x):
        """
        Args:
            x: [B, C, 256, 256] multi-channel input
               C=3: [original, CLAHE, gradient]

        Returns:
            x_fused: [B, C, 256, 256] reconstructed output
            anomaly_map: [B, 1, 256, 256] detection map
            fusion_weights: [B, 1, 256, 256] how much each stream contributes
        """
        # Extract features
        features = self.feature_extractor(x)

        # Score anomaly (per-pixel)
        anomaly_map = self.anomaly_scorer(features)

        # Normal stream (identity-like)
        x_normal = self.normal_stream(x)

        # Abnormal stream (perfect reconstruction)
        x_abnormal = self.abnormal_stream(
            x,
            anomaly_map,
            self.anomaly_scorer.normal_prototypes
        )

        # Soft fusion
        x_fused, fusion_weights = self.fusion(
            x_normal,
            x_abnormal,
            anomaly_map
        )

        return x_fused, anomaly_map, fusion_weights

    def update_prototypes(self, x_normal):
        """
        Update normal prototypes with normal images

        Args:
            x_normal: [B, C, 256, 256] normal multi-channel images
        """
        with torch.no_grad():
            features = self.feature_extractor(x_normal)
            self.anomaly_scorer.update_prototypes(features['f4'])


if __name__ == "__main__":
    """Test"""
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        base_channels: int = 64
        num_prototypes: int = 1000
        feature_dim: int = 512

    config = TestConfig()
    model = DSRN(config)

    # Dummy input
    x = torch.randn(2, 1, 256, 256)

    # Forward
    x_fused, anomaly_map, fusion_weights = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {x_fused.shape}")
    print(f"Anomaly map shape: {anomaly_map.shape}")
    print(f"Fusion weights shape: {fusion_weights.shape}")

    # Update prototypes
    model.update_prototypes(x)
    print(f"\nPrototype count: {model.anomaly_scorer.prototype_count.item()}")

    print("\nDSRN test passed!")
