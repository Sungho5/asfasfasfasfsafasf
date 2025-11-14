"""
Feature Extractor: Shared multi-scale feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled  # Return both for skip connections


class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion"""

    def __init__(self, channels_list):
        super().__init__()

        self.channels_list = channels_list

        # Upsampling layers to match sizes
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, channels_list[0], 1),
                nn.BatchNorm2d(channels_list[0]),
                nn.ReLU(inplace=True)
            )
            for ch in channels_list
        ])

    def forward(self, features_dict):
        """
        Args:
            features_dict: Dict with keys 'f1', 'f2', 'f3', 'f4'

        Returns:
            fused: Fused multi-scale features
        """
        f1 = features_dict['f1']  # (B, 64, 256, 256)
        f2 = features_dict['f2']  # (B, 128, 128, 128)
        f3 = features_dict['f3']  # (B, 256, 64, 64)
        f4 = features_dict['f4']  # (B, 512, 32, 32)

        target_size = f1.shape[2:]  # (256, 256)

        # Upsample all to same size
        f2_up = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)

        # Project to same channels
        f1_proj = self.upsample_layers[0](f1)
        f2_proj = self.upsample_layers[1](f2_up)
        f3_proj = self.upsample_layers[2](f3_up)
        f4_proj = self.upsample_layers[3](f4_up)

        # Concatenate
        fused = torch.cat([f1_proj, f2_proj, f3_proj, f4_proj], dim=1)

        return fused


class FeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction
    정상/비정상 모두에 필요한 공통 feature 추출
    """

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)  # 256x256
        self.down1 = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)  # 128x128
        self.down2 = nn.MaxPool2d(2, 2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)  # 64x64
        self.down3 = nn.MaxPool2d(2, 2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)  # 32x32

        # Multi-scale fusion
        self.fusion = MultiScaleFusion([
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8
        ])

        print(f"[FeatureExtractor] Initialized with base_channels={base_channels}")

    def forward(self, x):
        """
        Args:
            x: Input image (B, 1, 256, 256)

        Returns:
            features: Dict with keys 'f1', 'f2', 'f3', 'f4', 'fused'
        """
        # Encoder
        f1 = self.enc1(x)  # (B, 64, 256, 256)
        x = self.down1(f1)

        f2 = self.enc2(x)  # (B, 128, 128, 128)
        x = self.down2(f2)

        f3 = self.enc3(x)  # (B, 256, 64, 64)
        x = self.down3(f3)

        f4 = self.enc4(x)  # (B, 512, 32, 32)

        # Multi-scale fusion
        features_dict = {
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'f4': f4
        }

        fused = self.fusion(features_dict)
        features_dict['fused'] = fused

        return features_dict


if __name__ == "__main__":
    """Test"""

    model = FeatureExtractor(in_channels=1, base_channels=64)

    # Dummy input
    x = torch.randn(2, 1, 256, 256)

    # Forward
    features = model(x)

    print("\nFeature shapes:")
    for key, val in features.items():
        print(f"  {key}: {val.shape}")

    # Expected:
    # f1: (2, 64, 256, 256)
    # f2: (2, 128, 128, 128)
    # f3: (2, 256, 64, 64)
    # f4: (2, 512, 32, 32)
    # fused: (2, 256, 256, 256)  # 64*4 channels

    print("\nFeatureExtractor test passed!")
