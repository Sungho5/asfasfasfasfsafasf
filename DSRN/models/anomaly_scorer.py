"""
Spatial Anomaly Scorer
각 위치가 얼마나 비정상인지 점수화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAnomalyScorer(nn.Module):
    """
    각 위치의 anomaly score 계산

    Output: [B, 1, 256, 256] in [0, 1]
    0 = 완전 정상 (bypass)
    1 = 완전 비정상 (reconstruct)
    """

    def __init__(self, num_prototypes=1000, feature_dim=512):
        super().__init__()

        # Normal prototypes (학습 중 업데이트)
        self.register_buffer(
            'normal_prototypes',
            torch.zeros(num_prototypes, feature_dim)
        )
        self.register_buffer(
            'prototype_count',
            torch.zeros(1, dtype=torch.long)
        )

        # Scorer network (learned)
        self.scorer = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, 1, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

        # Multi-scale aggregation
        self.aggregate = nn.ModuleDict({
            'scale_1': nn.Conv2d(1, 1, 3, 1, 1),
            'scale_2': nn.Conv2d(1, 1, 5, 1, 2),
            'scale_3': nn.Conv2d(1, 1, 7, 1, 3),
        })

        print(f"[AnomalyScorer] Created with {num_prototypes} prototypes, {feature_dim}D features")

    def update_prototypes(self, features, max_prototypes=1000):
        """
        정상 이미지의 feature로 prototype 업데이트

        Args:
            features: [B, C, H, W] from normal images
            max_prototypes: Maximum number of prototypes
        """
        with torch.no_grad():
            B, C, H, W = features.shape

            # Spatial average pooling
            proto = features.mean(dim=[2, 3])  # [B, C]

            # Add to buffer
            current_count = self.prototype_count.item()

            for i in range(B):
                if current_count < max_prototypes:
                    self.normal_prototypes[current_count] = proto[i]
                    current_count += 1
                else:
                    # Replace random prototype
                    idx = torch.randint(0, max_prototypes, (1,)).item()
                    self.normal_prototypes[idx] = proto[i]

            self.prototype_count[0] = min(current_count, max_prototypes)

    def forward(self, features):
        """
        Args:
            features: dict with f4 [B, 512, 32, 32]

        Returns:
            anomaly_map: [B, 1, 256, 256]
        """
        f4 = features['f4']  # [B, 512, 32, 32]
        B, C, H, W = f4.shape

        # === Distance-based anomaly score ===
        if self.prototype_count > 0:
            # Flatten spatial dimensions
            f4_flat = f4.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

            # Get valid prototypes
            num_protos = self.prototype_count.item()
            prototypes = self.normal_prototypes[:num_protos]  # [N, C]

            # Compute pairwise distances
            distances = torch.cdist(
                f4_flat.unsqueeze(0),  # [1, B*H*W, C]
                prototypes.unsqueeze(0)  # [1, N, C]
            )  # [1, B*H*W, N]

            # Min distance to nearest normal prototype
            min_dist = distances.min(dim=-1)[0]  # [1, B*H*W]
            min_dist = min_dist.reshape(B, H, W).unsqueeze(1)  # [B, 1, H, W]

            # Normalize to [0, 1] using sigmoid
            anomaly_score_dist = torch.sigmoid(min_dist - 2.0)
        else:
            # No prototypes yet
            anomaly_score_dist = torch.zeros(B, 1, H, W, device=f4.device)

        # === Learned anomaly score ===
        anomaly_score_learned = self.scorer(f4)  # [B, 1, H, W]

        # === Combine ===
        anomaly_score = (anomaly_score_dist + anomaly_score_learned) / 2

        # === Multi-scale aggregation ===
        s1 = self.aggregate['scale_1'](anomaly_score)
        s2 = self.aggregate['scale_2'](anomaly_score)
        s3 = self.aggregate['scale_3'](anomaly_score)

        anomaly_score = (s1 + s2 + s3) / 3

        # Clamp to [0, 1] for BCE loss compatibility
        anomaly_score = torch.clamp(anomaly_score, 0.0, 1.0)

        # === Upsample to full resolution ===
        anomaly_map = F.interpolate(
            anomaly_score,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )

        return anomaly_map


if __name__ == "__main__":
    """Test"""

    scorer = SpatialAnomalyScorer(num_prototypes=100, feature_dim=512)

    # Dummy features
    features = {
        'f4': torch.randn(2, 512, 32, 32)
    }

    # Update prototypes with normal data
    normal_features = torch.randn(4, 512, 32, 32)
    scorer.update_prototypes(normal_features)

    print(f"Prototype count: {scorer.prototype_count.item()}")

    # Forward
    anomaly_map = scorer(features)

    print(f"Anomaly map shape: {anomaly_map.shape}")
    print(f"Anomaly map range: [{anomaly_map.min():.3f}, {anomaly_map.max():.3f}]")

    print("\nAnomalyScorer test passed!")
