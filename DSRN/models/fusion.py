"""
Soft Fusion Module
anomaly_map에 따라 두 stream을 부드럽게 혼합
"""

import torch
import torch.nn as nn


class SoftFusion(nn.Module):
    """
    anomaly_map에 따라 두 stream soft mixing

    anomaly_map = 0 → 100% normal stream
    anomaly_map = 1 → 100% abnormal stream
    anomaly_map = 0.5 → 50-50 mix
    """

    def __init__(self):
        super().__init__()

        # Refinement of fusion weights
        self.weight_refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        print("[SoftFusion] Created")

    def forward(self, x_normal, x_abnormal, anomaly_map):
        """
        Args:
            x_normal: [B, 1, 256, 256] from normal stream
            x_abnormal: [B, 1, 256, 256] from abnormal stream
            anomaly_map: [B, 1, 256, 256] selection weights

        Returns:
            x_fused: [B, 1, 256, 256] final output
            fusion_weights: [B, 1, 256, 256] refined weights
        """
        # Refine weights
        w = self.weight_refine(anomaly_map)  # [B, 1, 256, 256]

        # Soft selection
        x_fused = (1 - w) * x_normal + w * x_abnormal

        return x_fused, w


if __name__ == "__main__":
    """Test"""

    fusion = SoftFusion()

    # Dummy inputs
    x_normal = torch.randn(2, 1, 256, 256)
    x_abnormal = torch.randn(2, 1, 256, 256)
    anomaly_map = torch.rand(2, 1, 256, 256)

    # Forward
    x_fused, fusion_weights = fusion(x_normal, x_abnormal, anomaly_map)

    print(f"Normal stream shape: {x_normal.shape}")
    print(f"Abnormal stream shape: {x_abnormal.shape}")
    print(f"Anomaly map shape: {anomaly_map.shape}")
    print(f"Fused output shape: {x_fused.shape}")
    print(f"Fusion weights shape: {fusion_weights.shape}")
    print(f"Fusion weights range: [{fusion_weights.min():.3f}, {fusion_weights.max():.3f}]")

    print("\nSoftFusion test passed!")
