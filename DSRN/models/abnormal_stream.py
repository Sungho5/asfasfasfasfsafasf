"""
Abnormal Stream - Perfect Reconstruction
비정상 부분을 완전히 정상으로 재구성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureSynthesizer(nn.Module):
    """
    정상 texture를 생성하는 핵심 모듈
    Attention to normal prototypes
    """

    def __init__(self, feature_dim=512, attention_dim=256):
        super().__init__()

        # Attention components
        self.query = nn.Conv2d(feature_dim, attention_dim, 1)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # Texture generator
        self.gen = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
        )

        self.scale = attention_dim ** -0.5

    def forward(self, features, prototypes):
        """
        Args:
            features: [B, 512, H, W] 현재 feature
            prototypes: [N, 512] 정상 prototype features

        Returns:
            texture: [B, 256, H, W] 생성된 정상 texture
        """
        B, C, H, W = features.shape
        N = prototypes.shape[0]

        if N == 0:
            # No prototypes yet, return zeros
            return torch.zeros(B, 256, H, W, device=features.device)

        # Query from current features
        q = self.query(features)  # [B, 256, H, W]
        q = q.permute(0, 2, 3, 1).reshape(B * H * W, -1)  # [B*H*W, 256]

        # Key/Value from prototypes
        k = self.key(prototypes)  # [N, 256]
        v = self.value(prototypes)  # [N, 512]

        # Attention
        attn = torch.matmul(q, k.t()) * self.scale  # [B*H*W, N]
        attn = F.softmax(attn, dim=-1)

        # Aggregate
        texture_feat = torch.matmul(attn, v)  # [B*H*W, 512]
        texture_feat = texture_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Generate texture
        texture = self.gen(texture_feat)

        return texture


class AbnormalStream(nn.Module):
    """
    비정상 부분을 PERFECT 정상으로 재구성

    핵심: 주변 정상 context를 활용한 생성
    """

    def __init__(self):
        super().__init__()

        # Context encoder (주변 정상 영역 분석)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(2, 64, 7, 1, 3),  # input + anomaly_map
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )

        # Reconstruction decoder - Coarse
        self.coarse = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )

        # Texture synthesizer (정상 texture 생성)
        self.texture_gen = TextureSynthesizer()

        # Reconstruction decoder - Fine
        self.fine = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),  # 256 + 256 from texture
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
        )

        print("[AbnormalStream] Created")

    def forward(self, x, anomaly_map, normal_prototypes):
        """
        Args:
            x: [B, 1, 256, 256] input
            anomaly_map: [B, 1, 256, 256] where to reconstruct
            normal_prototypes: [N, 512] 정상 texture features

        Returns:
            x_recon: [B, 1, 256, 256] perfectly reconstructed
        """
        # Concatenate input with anomaly map
        x_with_mask = torch.cat([x, anomaly_map], dim=1)  # [B, 2, 256, 256]

        # Encode context (정상 영역 정보)
        context = self.context_encoder(x_with_mask)  # [B, 256, 256, 256]

        # Coarse reconstruction
        h_coarse = self.coarse(context)  # [B, 256, 256, 256]

        # Downsample for texture synthesis (32x32)
        h_down = F.avg_pool2d(h_coarse, kernel_size=8, stride=8)  # [B, 256, 32, 32]

        # Convert to 512 channels for texture synthesizer
        h_down_512 = F.conv2d(
            h_down,
            weight=torch.randn(512, 256, 1, 1, device=h_down.device) * 0.02,
            bias=None
        )  # [B, 512, 32, 32]

        # Generate normal texture based on prototypes
        texture_down = self.texture_gen(h_down_512, normal_prototypes)  # [B, 256, 32, 32]

        # Upsample texture back to full resolution
        texture = F.interpolate(
            texture_down,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )  # [B, 256, 256, 256]

        # Combine coarse reconstruction with texture
        combined = torch.cat([h_coarse, texture], dim=1)  # [B, 512, 256, 256]

        # Fine reconstruction
        x_recon = self.fine(combined)  # [B, 1, 256, 256]

        return x_recon


if __name__ == "__main__":
    """Test"""

    stream = AbnormalStream()

    # Dummy inputs
    x = torch.randn(2, 1, 256, 256)
    anomaly_map = torch.rand(2, 1, 256, 256)
    prototypes = torch.randn(100, 512)

    # Forward
    x_recon = stream(x, anomaly_map, prototypes)

    print(f"Input shape: {x.shape}")
    print(f"Anomaly map shape: {anomaly_map.shape}")
    print(f"Prototypes shape: {prototypes.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")

    print("\nAbnormalStream test passed!")
