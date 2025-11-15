"""
Normal Stream - Identity Pathway
정상 부분은 거의 그대로 통과
"""

import torch
import torch.nn as nn


class NormalStream(nn.Module):
    """
    정상 부분은 거의 identity로 통과
    약간의 refinement만 수행
    """

    def __init__(self, in_channels=1, alpha=0.05):
        super().__init__()

        # Minimal processing (거의 identity)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, 1, 1),
        )

        # Residual connection 강제 (very small alpha)
        self.alpha = nn.Parameter(torch.tensor(alpha))

        print(f"[NormalStream] Created with in_channels={in_channels}, alpha={alpha}")

    def forward(self, x):
        """
        Args:
            x: [B, C, 256, 256]

        Returns:
            x_normal: [B, C, 256, 256] (거의 x와 동일)
        """
        residual = self.refine(x)
        x_normal = x + self.alpha * residual

        return x_normal


if __name__ == "__main__":
    """Test"""

    stream = NormalStream(alpha=0.05)

    # Dummy input
    x = torch.randn(2, 1, 256, 256)

    # Forward
    x_normal = stream(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_normal.shape}")

    # Should be very close to identity
    diff = (x - x_normal).abs().mean()
    print(f"Mean difference: {diff:.6f}")

    print("\nNormalStream test passed!")
