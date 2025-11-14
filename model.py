import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from typing import Tuple, Dict
import numpy as np
import torchvision.models as models


class VAEFineTuner(nn.Module):
    def __init__(
            self,
            pretrained_model: str = "stabilityai/sd-vae-ft-mse",
            freeze_encoder: bool = True,
            unfreeze_last_n_blocks: int = 0,  # NEW
    ):
        super().__init__()

        # Load pretrained VAE
        self.vae = AutoencoderKL.from_pretrained(pretrained_model)

        # Freeze encoder if specified
        if freeze_encoder:
            # Freeze all encoder first
            for param in self.vae.encoder.parameters():
                param.requires_grad = False

            # Unfreeze last N down_blocks + mid_block
            if unfreeze_last_n_blocks > 0:
                # Get encoder blocks
                down_blocks = list(self.vae.encoder.down_blocks)

                # Unfreeze last N blocks
                for block in down_blocks[-unfreeze_last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

                # Always unfreeze mid_block (bottleneck) when unfreezing encoder
                if hasattr(self.vae.encoder, 'mid_block'):
                    for param in self.vae.encoder.mid_block.parameters():
                        param.requires_grad = True

                print(f"[Model] Encoder: Unfroze last {unfreeze_last_n_blocks} down_blocks + mid_block")
            else:
                print("[Model] Encoder: Completely frozen")
        else:
            print("[Model] Both Encoder and Decoder will be fine-tuned")

        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[Model] Total params: {total_params:,} | Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, 3, H, W) input images in [-1, 1]

        Returns:
            x_rec: (B, 3, H, W) reconstructed images
            posterior: dict with 'latent', 'mean', 'logvar'
        """
        # Encode
        posterior = self.vae.encode(x).latent_dist

        # Sample latent (for training) or use mean (for inference)
        if self.training:
            z = posterior.sample()
        else:
            z = posterior.mode()  # Use mean during eval

        # Decode
        x_rec = self.vae.decode(z).sample

        return x_rec, {
            'latent': z,
            'mean': posterior.mean,
            'logvar': posterior.logvar,
        }

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: (B, 3, H, W)
            sample: If True, sample from posterior. If False, use mean.

        Returns:
            z: (B, 4, H/8, W/8) latent
        """
        posterior = self.vae.encode(x).latent_dist

        if sample:
            return posterior.sample()
        else:
            return posterior.mode()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image space.

        Args:
            z: (B, 4, H/8, W/8)

        Returns:
            x_rec: (B, 3, H, W)
        """
        return self.vae.decode(z).sample


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for texture preservation.
    Uses multiple layers of VGG16 to capture multi-scale features.
    """

    def __init__(self, layers=[2, 7, 12, 21, 30], weights=None):
        """
        Args:
            layers: Indices of VGG16 layers to use for feature extraction
            weights: Optional weights for each layer. If None, use uniform weights.
        """
        super().__init__()

        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.layers = layers

        # Layer weights (uniform if not specified)
        if weights is None:
            self.weights = [1.0] * len(layers)
        else:
            assert len(weights) == len(layers)
            self.weights = weights

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """Normalize from [-1, 1] to ImageNet stats"""
        # [-1, 1] → [0, 1]
        x = (x + 1.0) / 2.0
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y: (B, 3, H, W) in [-1, 1]

        Returns:
            perceptual_loss: scalar
        """
        # Normalize to ImageNet stats
        x = self._normalize(x)
        y = self._normalize(y)

        loss = 0.0
        layer_idx = 0

        # Extract features from multiple layers
        for i, layer in enumerate(self.vgg):  # 여기! self.vae → self.vgg
            x = layer(x)
            y = layer(y)

            if i in self.layers:
                # L1 loss on feature maps, weighted by layer importance
                loss += self.weights[layer_idx] * F.l1_loss(x, y)
                layer_idx += 1

        return loss / len(self.layers)  # Average across layers


class VAELoss(nn.Module):
    """
    Multi-term reconstruction loss for VAE fine-tuning.
    Includes: L1, Gradient, HF (Laplacian), SSIM, Perceptual, KL
    """

    def __init__(
            self,
            lambda_pix: float = 0.5,
            lambda_grad: float = 1.0,
            lambda_hf: float = 0.8,
            lambda_ssim: float = 0.3,
            lambda_percep: float = 1.0,  # NEW: Perceptual loss weight
            beta_kl: float = 1e-6,
    ):
        super().__init__()

        self.lambda_pix = lambda_pix
        self.lambda_grad = lambda_grad
        self.lambda_hf = lambda_hf
        self.lambda_ssim = lambda_ssim
        self.lambda_percep = lambda_percep
        self.beta_kl = beta_kl

        # Sobel filters for gradient
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

        # Laplacian filter for high-frequency
        self.register_buffer('laplacian', torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3))

        # Perceptual loss
        self.perceptual_loss = PerceptualLoss(
            layers=[2, 7, 12, 21, 30],  # Multiple VGG layers
            weights=[1.0, 1.0, 1.0, 1.0, 1.0]  # Equal weight
        )

        print(f"[Loss] Initialized with perceptual loss (lambda={lambda_percep})")

    def forward(
            self,
            x: torch.Tensor,
            x_rec: torch.Tensor,
            mean: torch.Tensor,
            logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: (B, 3, H, W) original
            x_rec: (B, 3, H, W) reconstructed
            mean: (B, 4, H/8, W/8) latent mean
            logvar: (B, 4, H/8, W/8) latent logvar

        Returns:
            total_loss: scalar
            loss_dict: individual loss components
        """
        # 1. Pixel loss (L1)
        loss_pix = F.l1_loss(x_rec, x)

        # 2. Gradient loss
        loss_grad = self._gradient_loss(x, x_rec)

        # 3. High-frequency (Laplacian) loss
        loss_hf = self._hf_loss(x, x_rec)

        # 4. SSIM loss (optional, 1 - SSIM)
        loss_ssim = 1.0 - self._ssim(x, x_rec)

        # 5. Perceptual loss (NEW)
        loss_percep = self.perceptual_loss(x, x_rec)

        # 6. KL divergence
        loss_kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        total_loss = (
                self.lambda_pix * loss_pix +
                self.lambda_grad * loss_grad +
                self.lambda_hf * loss_hf +
                self.lambda_ssim * loss_ssim +
                self.lambda_percep * loss_percep +  # NEW
                self.beta_kl * loss_kl
        )

        loss_dict = {
            'total': total_loss.item(),
            'pix': loss_pix.item(),
            'grad': loss_grad.item(),
            'hf': loss_hf.item(),
            'ssim': loss_ssim.item(),
            'percep': loss_percep.item(),  # NEW
            'kl': loss_kl.item(),
        }

        return total_loss, loss_dict

    def _gradient_loss(self, x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        """Gradient loss using Sobel filters"""
        # Convert to grayscale
        x_gray = x.mean(dim=1, keepdim=True)
        x_rec_gray = x_rec.mean(dim=1, keepdim=True)

        # Compute gradients
        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        grad_mag_x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        grad_x_rec = F.conv2d(x_rec_gray, self.sobel_x, padding=1)
        grad_y_rec = F.conv2d(x_rec_gray, self.sobel_y, padding=1)
        grad_mag_rec = torch.sqrt(grad_x_rec ** 2 + grad_y_rec ** 2 + 1e-8)

        return F.l1_loss(grad_mag_rec, grad_mag_x)

    def _hf_loss(self, x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        """High-frequency loss using Laplacian filter"""
        x_gray = x.mean(dim=1, keepdim=True)
        x_rec_gray = x_rec.mean(dim=1, keepdim=True)

        lap_x = F.conv2d(x_gray, self.laplacian, padding=1)
        lap_rec = F.conv2d(x_rec_gray, self.laplacian, padding=1)

        return F.l1_loss(lap_rec, lap_x)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Simplified SSIM computation"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Convert to grayscale
        x = x.mean(dim=1, keepdim=True)
        y = y.mean(dim=1, keepdim=True)

        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)

        sigma_x = F.avg_pool2d(x ** 2, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return ssim_map.mean()


# Test code
if __name__ == "__main__":
    model = VAEFineTuner(freeze_encoder=True)
    criterion = VAELoss()

    # Dummy input
    x = torch.randn(2, 3, 256, 256)

    # Forward
    x_rec, posterior = model(x)

    # Compute loss
    loss, loss_dict = criterion(x, x_rec, posterior['mean'], posterior['logvar'])

    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_rec.shape}")
    print(f"Latent shape: {posterior['latent'].shape}")
    print(f"Loss dict: {loss_dict}")
