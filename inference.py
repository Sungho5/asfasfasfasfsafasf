"""
Phase 4: Inference & Anomaly Detection
Normalizer: G(x) maps input X-ray to nearest normal X-ray.
Anomaly map: |x - G(x)| with multi-scale differences.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import VAEFineTuner
from flowdiff_model import FlowDiffUNet
from flowdiff_loss import NoiseSchedule
from config import get_config_phase4, DataConfig
from dataset import PanoramaXrayDataset


class Normalizer(nn.Module):
    """
    Normalizer: x → x̂ (nearest normal X-ray)

    Pipeline:
    1. Encode: x → z_enc (VAE encoder)
    2. Add noise: z_enc → z_t (at noise level t₀)
    3. Denoise: z_t → z̃ (Flow+Diff model)
    4. Decode: z̃ → x̂ (VAE decoder)
    """

    def __init__(
            self,
            vae: VAEFineTuner,
            flowdiff: FlowDiffUNet,
            noise_schedule: NoiseSchedule,
            noise_level: float = 0.4,
            num_denoising_steps: int = 1,
    ):
        """
        Args:
            vae: Fine-tuned VAE (frozen)
            flowdiff: Flow+Diff UNet (frozen)
            noise_schedule: Diffusion noise schedule
            noise_level: t₀ for adding noise (0~1)
            num_denoising_steps: Number of denoising iterations
        """
        super().__init__()
        self.vae = vae
        self.flowdiff = flowdiff
        self.noise_schedule = noise_schedule
        self.noise_level = noise_level
        self.num_denoising_steps = num_denoising_steps

        # Freeze all
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.flowdiff.parameters():
            param.requires_grad = False

        self.vae.eval()
        self.flowdiff.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input X-ray to nearest normal.

        Args:
            x: (B, 3, 256, 256) input X-ray in [-1, 1]

        Returns:
            x_normalized: (B, 3, 256, 256) normalized X-ray in [-1, 1]
        """
        # 1. Encode to latent
        z_enc = self.vae.encode(x, sample=False)  # (B, 4, 32, 32)

        # 2. Add noise at level t₀
        t_idx = int(self.noise_level * self.noise_schedule.num_timesteps)
        t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)

        sqrt_alpha_t, sqrt_one_minus_alpha_t = self.noise_schedule.get_alpha_values(t)
        eps = torch.randn_like(z_enc)
        z_t = sqrt_alpha_t * z_enc + sqrt_one_minus_alpha_t * eps

        # 3. Denoise (single-step or multi-step)
        z_denoised = self._denoise(z_t, t)

        # 4. Decode to image
        x_normalized = self.vae.decode(z_denoised)

        return x_normalized

    def _denoise(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Denoise latent using diffusion head.

        Args:
            z_t: (B, 4, 32, 32) noisy latent
            t: (B,) timestep indices

        Returns:
            z_0: (B, 4, 32, 32) denoised latent
        """
        if self.num_denoising_steps == 1:
            # Single-step denoising
            eps_pred, _ = self.flowdiff(z_t, t, return_both_heads=False)

            sqrt_alpha_t, sqrt_one_minus_alpha_t = self.noise_schedule.get_alpha_values(t)
            z_0 = (z_t - sqrt_one_minus_alpha_t * eps_pred) / (sqrt_alpha_t + 1e-8)

            return z_0
        else:
            # Multi-step denoising (DDIM-like)
            # TODO: Implement if needed
            raise NotImplementedError("Multi-step denoising not implemented yet")


class AnomalyDetector:
    """
    Anomaly detection using Normalizer.

    Anomaly map: A = α|x - x̂| + β|∇x - ∇x̂| + γ|Lap(x) - Lap(x̂)|
    """

    def __init__(
            self,
            normalizer: Normalizer,
            alpha_l1: float = 1.0,
            beta_grad: float = 0.5,
            gamma_hf: float = 0.3,
    ):
        self.normalizer = normalizer
        self.alpha_l1 = alpha_l1
        self.beta_grad = beta_grad
        self.gamma_hf = gamma_hf

        # Sobel filters
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Laplacian
        self.laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

    @torch.no_grad()
    def detect(
            self,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Detect anomalies in input X-ray.

        Args:
            x: (B, 3, 256, 256) input X-ray in [-1, 1]

        Returns:
            x_normalized: (B, 3, 256, 256) normalized version
            anomaly_map: (B, 1, 256, 256) anomaly heatmap
            components: dict with individual components
        """
        device = x.device

        # Move filters to device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.laplacian = self.laplacian.to(device)

        # Normalize input
        x_norm = self.normalizer(x)  # (B, 3, 256, 256)

        # Compute residual components
        # 1. L1 difference
        diff_l1 = torch.abs(x - x_norm).mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # 2. Gradient difference
        diff_grad = self._gradient_diff(x, x_norm)

        # 3. High-frequency difference
        diff_hf = self._hf_diff(x, x_norm)

        # Combine into anomaly map
        anomaly_map = (
                self.alpha_l1 * diff_l1 +
                self.beta_grad * diff_grad +
                self.gamma_hf * diff_hf
        )

        components = {
            'l1': diff_l1,
            'grad': diff_grad,
            'hf': diff_hf,
        }

        return x_norm, anomaly_map, components

    def _gradient_diff(self, x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        """Gradient magnitude difference"""
        x_gray = x.mean(dim=1, keepdim=True)
        x_rec_gray = x_rec.mean(dim=1, keepdim=True)

        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        grad_mag_x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        grad_x_rec = F.conv2d(x_rec_gray, self.sobel_x, padding=1)
        grad_y_rec = F.conv2d(x_rec_gray, self.sobel_y, padding=1)
        grad_mag_rec = torch.sqrt(grad_x_rec ** 2 + grad_y_rec ** 2 + 1e-8)

        return torch.abs(grad_mag_x - grad_mag_rec)

    def _hf_diff(self, x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        """High-frequency (Laplacian) difference"""
        x_gray = x.mean(dim=1, keepdim=True)
        x_rec_gray = x_rec.mean(dim=1, keepdim=True)

        lap_x = F.conv2d(x_gray, self.laplacian, padding=1)
        lap_rec = F.conv2d(x_rec_gray, self.laplacian, padding=1)

        return torch.abs(lap_x - lap_rec)


def visualize_result(
        x_orig: torch.Tensor,
        x_norm: torch.Tensor,
        anomaly_map: torch.Tensor,
        components: dict,
        save_path: Optional[Path] = None,
):
    """
    Visualize original, normalized, and anomaly map.

    Args:
        x_orig: (3, 256, 256) original image
        x_norm: (3, 256, 256) normalized image
        anomaly_map: (1, 256, 256) anomaly heatmap
        components: dict with individual difference components
        save_path: Optional path to save figure
    """
    # Convert to numpy and denormalize
    x_orig = (x_orig.cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
    x_norm = (x_norm.cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
    anomaly_map = anomaly_map.cpu().numpy()[0]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original, Normalized, Residual
    axes[0, 0].imshow(x_orig, cmap='gray')
    axes[0, 0].set_title('Original X-ray')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(x_norm, cmap='gray')
    axes[0, 1].set_title('Normalized (G(x))')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.abs(x_orig - x_norm), cmap='hot')
    axes[0, 2].set_title('Pixel Residual |x - G(x)|')
    axes[0, 2].axis('off')

    # Row 2: Component anomaly maps
    l1_map = components['l1'].cpu().numpy()[0]
    grad_map = components['grad'].cpu().numpy()[0]
    hf_map = components['hf'].cpu().numpy()[0]

    axes[1, 0].imshow(l1_map, cmap='hot')
    axes[1, 0].set_title('L1 Component')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(grad_map, cmap='hot')
    axes[1, 1].set_title('Gradient Component')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(anomaly_map, cmap='hot')
    axes[1, 2].set_title('Final Anomaly Map')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Save] Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Test inference on a few samples"""
    cfg = get_config_phase4()
    data_cfg = DataConfig()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}")

    # ===== Load VAE =====
    print(f"[Model] Loading VAE from {cfg.vae_checkpoint}")
    vae = VAEFineTuner(freeze_encoder=True)
    vae_ckpt = torch.load(cfg.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.to(device).eval()

    # ===== Load Flow+Diff =====
    print(f"[Model] Loading Flow+Diff from {cfg.flowdiff_checkpoint}")
    flowdiff = FlowDiffUNet(use_dual_head=True)
    flowdiff_ckpt = torch.load(cfg.flowdiff_checkpoint, map_location=device)
    flowdiff.load_state_dict(flowdiff_ckpt['model_state_dict'])
    flowdiff.to(device).eval()

    # ===== Noise Schedule =====
    noise_schedule = NoiseSchedule(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type='linear',
    )

    # ===== Create Normalizer =====
    normalizer = Normalizer(
        vae=vae,
        flowdiff=flowdiff,
        noise_schedule=noise_schedule,
        noise_level=cfg.noise_level,
        num_denoising_steps=cfg.num_denoising_steps,
    )

    # ===== Create Anomaly Detector =====
    detector = AnomalyDetector(
        normalizer=normalizer,
        alpha_l1=cfg.alpha_l1,
        beta_grad=cfg.beta_grad,
        gamma_hf=cfg.gamma_hf,
    )

    # ===== Load Test Dataset =====
    # For demo, use validation set
    from dataset import create_train_val_split

    _, val_files = create_train_val_split(
        data_dir=data_cfg.data_dir,
        val_ratio=data_cfg.val_ratio,
        seed=data_cfg.seed,
    )

    val_dataset = PanoramaXrayDataset(
        data_dir=data_cfg.data_dir,
        file_list=val_files,
        window_center=data_cfg.window_center,
        window_width=data_cfg.window_width,
        target_size=data_cfg.target_size,
        augment=False,
    )

    print(f"[Dataset] Loaded {len(val_dataset)} validation samples")

    # ===== Run Inference =====
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(10, len(val_dataset))
    print(f"\n[Inference] Processing {num_samples} samples...")

    for idx in tqdm(range(num_samples)):
        # Load image
        x = val_dataset[idx].unsqueeze(0).to(device)  # (1, 3, 256, 256)

        # Detect anomalies
        x_norm, anomaly_map, components = detector.detect(x)

        # Visualize
        save_path = output_dir / f"result_{idx:03d}.png"
        visualize_result(
            x_orig=x[0],
            x_norm=x_norm[0],
            anomaly_map=anomaly_map[0],
            components={k: v[0] for k, v in components.items()},
            save_path=save_path,
        )

    print(f"\n[Complete] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
