"""
Flow+Diffusion UNet Model
Unified UNet backbone with dual heads for:
1. Diffusion: noise prediction (epsilon)
2. Flow Matching: velocity prediction (v)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


# ========== Time Embedding ==========

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for timesteps.
    Maps timestep t to a high-dimensional vector.
    """

    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

        # MLP to project embedding
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of timestep indices or continuous values

        Returns:
            emb: (B, embedding_dim * 4) time embedding
        """
        # Sinusoidal embedding
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, embedding_dim)

        # MLP projection
        emb = self.mlp(emb)  # (B, embedding_dim * 4)

        return emb


# ========== Residual Block ==========

class ResBlock(nn.Module):
    """
    Residual block with time embedding injection.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_emb_dim: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First conv
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        # Time embedding projection
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # Second conv
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            time_emb: (B, time_emb_dim)

        Returns:
            out: (B, out_channels, H, W)
        """
        h = self.conv1(x)

        # Add time embedding (broadcast to spatial dimensions)
        time_emb = self.time_emb_proj(time_emb)[:, :, None, None]  # (B, C, 1, 1)
        h = h + time_emb

        h = self.conv2(h)

        # Skip connection
        return h + self.skip(x)


# ========== Attention ==========

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for spatial features"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

        # Normalize
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x)  # (B, 3*C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)  # (B, 3, num_heads, head_dim, H*W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)  # (B, C, H, W)

        # Output projection
        out = self.proj(out)

        return out + residual


# ========== Downsample / Upsample ==========

class Downsample(nn.Module):
    """Downsampling layer (2x)"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer (2x)"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ========== UNet with Dual Head ==========

class FlowDiffUNet(nn.Module):
    """
    UNet backbone with dual heads for Flow + Diffusion.

    Architecture:
        Input: (B, 4, 32, 32) latent
        → Down blocks + Attention
        → Mid block
        → Up blocks + Skip connections
        → Dual heads:
            - Diffusion head: predicts noise ε
            - Flow head: predicts velocity v
    """

    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 4,
            model_channels: int = 128,
            num_res_blocks: int = 2,
            attention_resolutions: Tuple[int, ...] = (8, 16),
            channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
            dropout: float = 0.0,
            num_heads: int = 4,
            use_dual_head: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.use_dual_head = use_dual_head

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(model_channels, max_period=10000)

        # Input conv
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = model_channels
        input_block_channels = [model_channels]

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for block_idx in range(num_res_blocks):
                layers = [ResBlock(current_channels, out_ch, time_emb_dim, dropout)]

                # Add attention at specific resolutions
                # Resolution at this level: 32 / 2^level
                current_res = 32 // (2 ** level)
                if current_res in attention_resolutions:
                    layers.append(MultiHeadAttention(out_ch, num_heads))

                self.down_blocks.append(nn.ModuleList(layers))
                current_channels = out_ch
                input_block_channels.append(current_channels)

            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.downsamples.append(Downsample(current_channels))
                input_block_channels.append(current_channels)
            else:
                self.downsamples.append(None)

        # Mid block
        self.mid_block = nn.ModuleList([
            ResBlock(current_channels, current_channels, time_emb_dim, dropout),
            MultiHeadAttention(current_channels, num_heads),
            ResBlock(current_channels, current_channels, time_emb_dim, dropout),
        ])

        # Up blocks
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult

            for block_idx in range(num_res_blocks + 1):
                # Account for skip connection from down block
                skip_ch = input_block_channels.pop()
                layers = [ResBlock(current_channels + skip_ch, out_ch, time_emb_dim, dropout)]

                # Add attention
                current_res = 32 // (2 ** level)
                if current_res in attention_resolutions:
                    layers.append(MultiHeadAttention(out_ch, num_heads))

                self.up_blocks.append(nn.ModuleList(layers))
                current_channels = out_ch

            # Upsample (except first level when going backwards)
            if level > 0:
                self.upsamples.append(Upsample(current_channels))
            else:
                self.upsamples.append(None)

        # Output heads
        if use_dual_head:
            # Separate heads for diffusion and flow
            self.diffusion_head = nn.Sequential(
                nn.GroupNorm(32, current_channels),
                nn.SiLU(),
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            )

            self.flow_head = nn.Sequential(
                nn.GroupNorm(32, current_channels),
                nn.SiLU(),
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            # Single head (for baseline experiments)
            self.output_head = nn.Sequential(
                nn.GroupNorm(32, current_channels),
                nn.SiLU(),
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            )

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            return_both_heads: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 4, 32, 32) latent input
            timesteps: (B,) timestep indices or continuous values in [0, 1]
            return_both_heads: If True, return (diffusion_out, flow_out). Else, just diffusion_out.

        Returns:
            If use_dual_head and return_both_heads:
                (diffusion_out, flow_out): both (B, 4, 32, 32)
            Else:
                diffusion_out: (B, 4, 32, 32)
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)  # (B, time_emb_dim)

        # Input
        h = self.input_conv(x)  # (B, model_channels, 32, 32)

        # Down blocks (with skip connections storage)
        skips = [h]

        for down_block, downsample in zip(self.down_blocks, self.downsamples):
            for layer in down_block:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb)
                else:  # Attention
                    h = layer(h)

            skips.append(h)

            if downsample is not None:
                h = downsample(h)
                skips.append(h)

        # Mid block
        for layer in self.mid_block:
            if isinstance(layer, ResBlock):
                h = layer(h, time_emb)
            else:  # Attention
                h = layer(h)

        # Up blocks (with skip connections)
        for up_block, upsample in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for layer in up_block:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb)
                else:  # Attention
                    h = layer(h)

            if upsample is not None:
                h = upsample(h)

        # Output heads
        if self.use_dual_head:
            diffusion_out = self.diffusion_head(h)
            flow_out = self.flow_head(h)

            if return_both_heads:
                return diffusion_out, flow_out
            else:
                return diffusion_out, None
        else:
            out = self.output_head(h)
            return out, None


# ========== EMA Model Wrapper ==========

class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Used for stable inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Apply EMA parameters to model (for inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

        self.backup = {}


# ========== Test Code ==========

if __name__ == "__main__":
    # Test UNet
    model = FlowDiffUNet(
        in_channels=4,
        out_channels=4,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 4),
        dropout=0.0,
        num_heads=4,
        use_dual_head=True,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 32, 32)
    t = torch.rand(batch_size) * 1000  # Random timesteps

    # Single head output
    diff_out, flow_out = model(x, t, return_both_heads=False)
    print(f"\nInput shape: {x.shape}")
    print(f"Diffusion output shape: {diff_out.shape}")
    print(f"Flow output: {flow_out}")

    # Dual head output
    diff_out, flow_out = model(x, t, return_both_heads=True)
    print(f"\nDual head mode:")
    print(f"Diffusion output shape: {diff_out.shape}")
    print(f"Flow output shape: {flow_out.shape}")

    # Test EMA
    ema = EMAModel(model, decay=0.9999)
    ema.update()
    print(f"\nEMA initialized and updated")
