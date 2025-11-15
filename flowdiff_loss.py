"""
Loss Functions for Flow + Diffusion Training
Includes:
1. Diffusion loss (noise prediction)
2. Flow matching loss (velocity prediction)
3. Identity constraint loss (normal -> normal preservation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


# ========== Noise Schedule ==========

class NoiseSchedule:
    """
    Diffusion noise schedule: defines β_t and α_t.
    """

    def __init__(
            self,
            num_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            schedule_type: str = 'linear',
    ):
        """
        Args:
            num_timesteps: Total number of diffusion steps
            beta_start: Starting β value
            beta_end: Ending β value
            schedule_type: 'linear' or 'cosine'
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Compute β schedule
        if schedule_type == 'linear':
            self.betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Compute α values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Precompute useful quantities
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def _cosine_schedule(self) -> np.ndarray:
        """Cosine noise schedule (more stable for high resolutions)"""
        s = 0.008
        steps = self.num_timesteps + 1
        x = np.linspace(0, self.num_timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / self.num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)

    def get_alpha_values(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sqrt(α_t) and sqrt(1 - α_t) for given timesteps.

        Args:
            t: (B,) timestep indices (0 to num_timesteps-1)

        Returns:
            sqrt_alpha_t: (B, 1, 1, 1)
            sqrt_one_minus_alpha_t: (B, 1, 1, 1)
        """
        # Convert to numpy indices
        t_np = t.cpu().numpy().astype(np.int32)
        t_np = np.clip(t_np, 0, self.num_timesteps - 1)

        sqrt_alpha = torch.from_numpy(self.sqrt_alphas_cumprod[t_np]).to(t.device)
        sqrt_one_minus_alpha = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod[t_np]).to(t.device)

        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        sqrt_alpha = sqrt_alpha.float().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.float().view(-1, 1, 1, 1)

        return sqrt_alpha, sqrt_one_minus_alpha


# ========== Diffusion Loss ==========

class DiffusionLoss(nn.Module):
    """
    Diffusion loss: train model to predict noise ε.

    Forward process: z_t = sqrt(α_t) * z_0 + sqrt(1 - α_t) * ε
    Objective: ||ε - ε_θ(z_t, t)||²
    """

    def __init__(self, noise_schedule: NoiseSchedule):
        super().__init__()
        self.noise_schedule = noise_schedule

    def forward(
            self,
            model: nn.Module,
            z_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model: FlowDiffUNet model
            z_0: (B, 4, 32, 32) clean latents

        Returns:
            loss: scalar
            info: dict with 'loss_diff'
        """
        B = z_0.shape[0]
        device = z_0.device

        # Sample random timesteps
        t = torch.randint(0, self.noise_schedule.num_timesteps, (B,), device=device)

        # Sample noise
        eps = torch.randn_like(z_0)

        # Add noise to z_0
        sqrt_alpha_t, sqrt_one_minus_alpha_t = self.noise_schedule.get_alpha_values(t)
        z_t = sqrt_alpha_t * z_0 + sqrt_one_minus_alpha_t * eps

        # Predict noise
        eps_pred, _ = model(z_t, t, return_both_heads=False)

        # MSE loss
        loss = F.mse_loss(eps_pred, eps)

        info = {'loss_diff': loss.item()}

        return loss, info


# ========== Flow Matching Loss ==========

class FlowMatchingLoss(nn.Module):
    """
    Flow matching loss: train model to predict velocity v.

    Interpolation: z_t = (1 - t) * z_base + t * z_data
    Target velocity: v* = z_data - z_base
    Objective: ||v_θ(z_t, t) - v*||²
    """

    def __init__(self, base_std: float = 1.0):
        """
        Args:
            base_std: Standard deviation of base noise distribution N(0, base_std²*I)
        """
        super().__init__()
        self.base_std = base_std

    def forward(
            self,
            model: nn.Module,
            z_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model: FlowDiffUNet model
            z_data: (B, 4, 32, 32) data latents (normal X-rays)

        Returns:
            loss: scalar
            info: dict with 'loss_flow'
        """
        B = z_data.shape[0]
        device = z_data.device

        # Sample base noise
        z_base = torch.randn_like(z_data) * self.base_std

        # Sample random time t ∈ [0, 1]
        t = torch.rand(B, device=device)

        # Interpolate: z_t = (1 - t) * z_base + t * z_data
        t_expanded = t.view(-1, 1, 1, 1)
        z_t = (1 - t_expanded) * z_base + t_expanded * z_data

        # Target velocity: v* = z_data - z_base
        v_target = z_data - z_base

        # Predict velocity (use flow head)
        _, v_pred = model(z_t, t, return_both_heads=True)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        info = {'loss_flow': loss.item()}

        return loss, info


# ========== Identity Constraint Loss ==========

class IdentityLoss(nn.Module):
    """
    Identity constraint: G(x) ≈ x for normal inputs.

    Pipeline: x → Encode → Add noise → Denoise → Decode → x̂
    Loss: ||x - x̂|| + λ_grad ||∇x - ∇x̂|| + λ_HF ||Lap(x) - Lap(x̂)||
    """

    def __init__(
            self,
            vae_model: nn.Module,
            noise_schedule: NoiseSchedule,
            noise_level: float = 0.4,
            lambda_pix: float = 1.0,
            lambda_grad: float = 0.5,
            lambda_hf: float = 0.3,
    ):
        """
        Args:
            vae_model: Fine-tuned VAE (frozen)
            noise_schedule: Diffusion noise schedule
            noise_level: Noise level t₀ ∈ [0, 1] for identity test
            lambda_pix: Weight for pixel loss
            lambda_grad: Weight for gradient loss
            lambda_hf: Weight for high-frequency loss
        """
        super().__init__()
        self.vae = vae_model
        self.noise_schedule = noise_schedule
        self.noise_level = noise_level
        self.lambda_pix = lambda_pix
        self.lambda_grad = lambda_grad
        self.lambda_hf = lambda_hf

        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # Sobel filters for gradient
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

        # Laplacian filter
        self.register_buffer('laplacian', torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3))

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent (use mean)"""
        return self.vae.encode(x, sample=False)

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        return self.vae.decode(z)

    def _denoise_one_step(
            self,
            model: nn.Module,
            z_t: torch.Tensor,
            t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-step denoising using diffusion head.

        Given z_t = sqrt(α_t) * z_0 + sqrt(1 - α_t) * ε,
        Predict ε̂ = ε_θ(z_t, t),
        Recover z_0 = (z_t - sqrt(1 - α_t) * ε̂) / sqrt(α_t)
        """
        # Predict noise
        eps_pred, _ = model(z_t, t, return_both_heads=False)

        # Get alpha values
        sqrt_alpha_t, sqrt_one_minus_alpha_t = self.noise_schedule.get_alpha_values(t)

        # Recover clean latent
        z_0_pred = (z_t - sqrt_one_minus_alpha_t * eps_pred) / (sqrt_alpha_t + 1e-8)

        return z_0_pred

    def forward(
            self,
            model: nn.Module,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model: FlowDiffUNet model
            x: (B, 3, 256, 256) normal X-ray images in [-1, 1]

        Returns:
            loss: scalar
            info: dict with loss components
        """
        B = x.shape[0]
        device = x.device

        # 1. Encode to latent
        z_enc = self._encode(x)  # (B, 4, 32, 32)

        # 2. Add noise at level t₀
        t_idx = int(self.noise_level * self.noise_schedule.num_timesteps)
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)

        sqrt_alpha_t, sqrt_one_minus_alpha_t = self.noise_schedule.get_alpha_values(t)
        eps = torch.randn_like(z_enc)
        z_t = sqrt_alpha_t * z_enc + sqrt_one_minus_alpha_t * eps

        # 3. Denoise with model
        z_denoised = self._denoise_one_step(model, z_t, t)

        # 4. Decode back to image
        x_rec = self._decode(z_denoised)  # (B, 3, 256, 256)

        # 5. Compute reconstruction losses
        # Pixel loss
        loss_pix = F.l1_loss(x_rec, x)

        # Gradient loss
        loss_grad = self._gradient_loss(x, x_rec)

        # High-frequency loss
        loss_hf = self._hf_loss(x, x_rec)

        # Total identity loss
        loss = (
                self.lambda_pix * loss_pix +
                self.lambda_grad * loss_grad +
                self.lambda_hf * loss_hf
        )

        info = {
            'loss_id': loss.item(),
            'loss_id_pix': loss_pix.item(),
            'loss_id_grad': loss_grad.item(),
            'loss_id_hf': loss_hf.item(),
        }

        return loss, info

    def _gradient_loss(self, x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        """Gradient magnitude loss"""
        x_gray = x.mean(dim=1, keepdim=True)
        x_rec_gray = x_rec.mean(dim=1, keepdim=True)

        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        grad_mag_x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        grad_x_rec = F.conv2d(x_rec_gray, self.sobel_x, padding=1)
        grad_y_rec = F.conv2d(x_rec_gray, self.sobel_y, padding=1)
        grad_mag_rec = torch.sqrt(grad_x_rec ** 2 + grad_y_rec ** 2 + 1e-8)

        return F.l1_loss(grad_mag_rec, grad_mag_x)

    def _hf_loss(self, x: torch.Tensor, x_rec: torch.Tensor) -> torch.Tensor:
        """High-frequency (Laplacian) loss"""
        x_gray = x.mean(dim=1, keepdim=True)
        x_rec_gray = x_rec.mean(dim=1, keepdim=True)

        lap_x = F.conv2d(x_gray, self.laplacian, padding=1)
        lap_rec = F.conv2d(x_rec_gray, self.laplacian, padding=1)

        return F.l1_loss(lap_rec, lap_x)


# ========== Combined Loss ==========

class FlowDiffLoss(nn.Module):
    """
    Combined loss for Flow + Diffusion training.

    Phase 1 (Generative): L_gen = λ_diff * L_diff + λ_flow * L_flow
    Phase 2 (+ Identity): L_total = L_gen + λ_id * L_id
    """

    def __init__(
            self,
            noise_schedule: NoiseSchedule,
            vae_model: Optional[nn.Module] = None,
            lambda_diff: float = 1.0,
            lambda_flow: float = 0.5,
            lambda_id_global: float = 0.1,
            identity_noise_level: float = 0.4,
            lambda_id_pix: float = 1.0,
            lambda_id_grad: float = 0.5,
            lambda_id_hf: float = 0.3,
    ):
        super().__init__()

        self.lambda_diff = lambda_diff
        self.lambda_flow = lambda_flow
        self.lambda_id_global = lambda_id_global

        # Sub-losses
        self.diffusion_loss = DiffusionLoss(noise_schedule)
        self.flow_loss = FlowMatchingLoss(base_std=1.0)

        # Identity loss (only if VAE provided)
        if vae_model is not None:
            self.identity_loss = IdentityLoss(
                vae_model=vae_model,
                noise_schedule=noise_schedule,
                noise_level=identity_noise_level,
                lambda_pix=lambda_id_pix,
                lambda_grad=lambda_id_grad,
                lambda_hf=lambda_id_hf,
            )
        else:
            self.identity_loss = None

    def forward(
            self,
            model: nn.Module,
            z_data: torch.Tensor,
            x_images: Optional[torch.Tensor] = None,
            use_identity: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model: FlowDiffUNet
            z_data: (B, 4, 32, 32) latent data
            x_images: (B, 3, 256, 256) optional images for identity loss
            use_identity: Whether to include identity constraint

        Returns:
            loss: scalar
            info: dict with all loss components
        """
        # Generative losses
        loss_diff, info_diff = self.diffusion_loss(model, z_data)
        loss_flow, info_flow = self.flow_loss(model, z_data)

        loss_gen = self.lambda_diff * loss_diff + self.lambda_flow * loss_flow

        info = {
            **info_diff,
            **info_flow,
            'loss_gen': loss_gen.item(),
        }

        # Add identity loss if enabled
        if use_identity and self.identity_loss is not None and x_images is not None:
            loss_id, info_id = self.identity_loss(model, x_images)
            total_loss = loss_gen + self.lambda_id_global * loss_id

            info.update(info_id)
            info['loss_total'] = total_loss.item()

            return total_loss, info
        else:
            info['loss_total'] = loss_gen.item()
            return loss_gen, info


# ========== Test Code ==========

if __name__ == "__main__":
    from flowdiff_model import FlowDiffUNet

    # Create noise schedule
    schedule = NoiseSchedule(num_timesteps=1000, schedule_type='linear')

    # Test alpha values
    t = torch.tensor([0, 100, 500, 999])
    sqrt_alpha, sqrt_one_minus = schedule.get_alpha_values(t)
    print(f"sqrt_alpha_t: {sqrt_alpha.squeeze()}")
    print(f"sqrt_one_minus_alpha_t: {sqrt_one_minus.squeeze()}")

    # Create model
    model = FlowDiffUNet(use_dual_head=True)

    # Test diffusion loss
    z = torch.randn(4, 4, 32, 32)
    diff_loss_fn = DiffusionLoss(schedule)
    loss_diff, info_diff = diff_loss_fn(model, z)
    print(f"\nDiffusion loss: {loss_diff.item():.4f}")

    # Test flow loss
    flow_loss_fn = FlowMatchingLoss()
    loss_flow, info_flow = flow_loss_fn(model, z)
    print(f"Flow loss: {loss_flow.item():.4f}")

    print("\nAll loss functions working correctly!")
