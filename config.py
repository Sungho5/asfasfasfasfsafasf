"""
Configuration for VAE + Flow + Diffusion pipeline
OPTIMIZED FOR SUBTLE TEXTURE/SHAPE DIFFERENCES (Tumor/Cyst in Dental X-rays)

Key features:
- CLAHE preprocessing for texture enhancement
- Morphological gradient as auxiliary supervision
- Texture-focused loss weights
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DataConfig:
    """Dataset configuration"""
    data_dir: str = '/home/imgadmin/DATA1/sungho/synology/sungho/NEW_Anomaly/data/2048/'
    val_ratio: float = 0.1
    window_center: int = 2000
    window_width: int = 4000
    target_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 42


@dataclass
class CLAHEConfig:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) settings"""
    enabled: bool = True
    clip_limit: float = 2.0      # Conservative to avoid over-enhancement
    tile_grid_size: int = 8      # 8x8 tiles for local adaptation


@dataclass
class MorphGradientConfig:
    """Morphological Gradient settings"""
    enabled: bool = True
    kernel_size: int = 3         # Elliptical kernel size (3x3)
    kernel_shape: str = 'ellipse'  # 'ellipse', 'rect', 'cross'

    # Multi-scale gradient
    use_multiscale: bool = True
    scales: Tuple[int, ...] = (3, 5)  # Kernel sizes
    scale_weights: Tuple[float, ...] = (0.6, 0.4)  # (fine, coarse)


@dataclass
class VAEConfig:
    """VAE fine-tuning configuration (Phase 1)

    OPTIMIZED FOR SUBTLE TEXTURE DETECTION:
    - High weight on HF loss (trabecular pattern)
    - High weight on morphological gradient (structural boundaries)
    - Lower weight on pixel intensity (subtle differences)
    """
    pretrained_model: str = 'stabilityai/sd-vae-ft-mse'
    freeze_encoder: bool = True
    unfreeze_last_n_blocks: int = 0

    # Auxiliary gradient head
    use_gradient_head: bool = True
    gradient_head_channels: Tuple[int, ...] = (512, 256, 128)

    # Loss weights - TEXTURE > SHAPE > INTENSITY
    lambda_pix: float = 0.5          # Pixel L1 (subtle intensity)
    lambda_grad: float = 0.5         # Sobel gradient (keep for compatibility)
    lambda_morph_grad: float = 1.5   # 🆕 Morphological gradient (HIGHEST!)
    lambda_hf: float = 1.0           # Laplacian (trabecular texture)
    lambda_ssim: float = 0.3         # Structural similarity
    lambda_percep: float = 0.8       # VGG perceptual (high-level texture)
    beta_kl: float = 1e-6            # KL divergence (keep latent stable)

    # CLAHE and morphological gradient configs
    clahe_config: CLAHEConfig = field(default_factory=CLAHEConfig)
    morph_grad_config: MorphGradientConfig = field(default_factory=MorphGradientConfig)

    # Training
    num_epochs: int = 50
    lr: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Checkpointing
    output_dir: str = './outputs/phase1_vae'
    save_every: int = 5

    # Wandb
    use_wandb: bool = False
    wandb_project: str = 'dental-xray-vae-texture'


@dataclass
class LatentConfig:
    """Latent dataset generation configuration (Phase 2)"""
    vae_checkpoint: str = './outputs/phase1_vae/checkpoint_best.pt'
    output_dir: str = './outputs/phase2_latents'
    batch_size: int = 16
    use_mean: bool = True


@dataclass
class DiffusionConfig:
    """Diffusion model configuration"""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: str = 'linear'
    prediction_type: str = 'epsilon'


@dataclass
class FlowConfig:
    """Flow matching configuration"""
    t_min: float = 0.0
    t_max: float = 1.0
    base_std: float = 1.0


@dataclass
class UNetConfig:
    """UNet backbone configuration"""
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (8, 16)
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    dropout: float = 0.0
    use_checkpoint: bool = False
    num_heads: int = 4
    use_dual_head: bool = True


@dataclass
class FlowDiffTrainingConfig:
    """Flow+Diff training configuration (Phase 3)

    OPTIMIZED FOR SUBTLE LESIONS:
    - Stronger identity constraint (preserve normal texture)
    - Lower noise level (preserve subtle information)
    - Texture-focused identity loss
    """
    # Data
    latent_dir: str = './outputs/phase2_latents'
    vae_checkpoint: str = './outputs/phase1_vae/checkpoint_best.pt'

    # Model
    unet_config: UNetConfig = field(default_factory=UNetConfig)
    diffusion_config: DiffusionConfig = field(default_factory=DiffusionConfig)
    flow_config: FlowConfig = field(default_factory=FlowConfig)

    # Curriculum learning
    phase1_steps: int = 100_000
    phase2_steps: int = 200_000
    total_steps: int = 300_000

    # Loss weights - Phase 1 (Generative)
    lambda_diff: float = 1.0
    lambda_flow: float = 0.5

    # Loss weights - Phase 2 (Identity) - STRONGER FOR SUBTLE DETECTION
    lambda_id_global: float = 0.2        # ↑ Stronger preservation
    identity_noise_level: float = 0.3    # ↓ Gentler (preserve subtle texture)
    identity_sample_ratio: float = 0.3

    # Identity components - TEXTURE FOCUSED
    lambda_id_pix: float = 0.5           # Intensity subtle
    lambda_id_grad: float = 1.0          # Boundary important
    lambda_id_hf: float = 1.5            # Texture critical

    # Warm-up
    identity_warmup_start: int = 90_000
    identity_warmup_end: int = 110_000

    # Training
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    ema_decay: float = 0.9999

    # Checkpointing
    output_dir: str = './outputs/phase3_flowdiff'
    save_every: int = 10_000
    val_every: int = 5_000

    # Wandb
    use_wandb: bool = False
    wandb_project: str = 'dental-xray-flowdiff-texture'


@dataclass
class InferenceConfig:
    """Inference configuration (Phase 4)

    OPTIMIZED FOR SUBTLE TEXTURE ANOMALIES:
    - Lower noise level (preserve subtle information)
    - High weight on HF difference (texture changes)
    - Moderate weight on gradient (boundary changes)
    """
    vae_checkpoint: str = './outputs/phase1_vae/checkpoint_best.pt'
    flowdiff_checkpoint: str = './outputs/phase3_flowdiff/checkpoint_best.pt'

    # Denoising - GENTLE for subtle lesions
    noise_level: float = 0.35            # ↓ Lower than before
    num_denoising_steps: int = 1
    use_ddim: bool = False

    # Anomaly map weights - TEXTURE > SHAPE > INTENSITY
    alpha_l1: float = 0.5                # Intensity subtle
    beta_grad: float = 1.0               # Boundary shape
    gamma_hf: float = 1.5                # Texture CRITICAL!

    # Multi-scale HF detection
    use_multiscale_hf: bool = True
    hf_scales: Tuple[int, ...] = (3, 5)
    hf_scale_weights: Tuple[float, ...] = (0.6, 0.4)

    # Output
    output_dir: str = './outputs/phase4_inference'
    save_heatmaps: bool = True
    save_comparisons: bool = True


def get_config_phase1() -> VAEConfig:
    """Get Phase 1 (VAE fine-tuning) config"""
    return VAEConfig()


def get_config_phase2() -> LatentConfig:
    """Get Phase 2 (Latent extraction) config"""
    return LatentConfig()


def get_config_phase3() -> FlowDiffTrainingConfig:
    """Get Phase 3 (Flow+Diff training) config"""
    return FlowDiffTrainingConfig()


def get_config_phase4() -> InferenceConfig:
    """Get Phase 4 (Inference) config"""
    return InferenceConfig()


@dataclass
class PipelineConfig:
    """Full pipeline configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    flowdiff: FlowDiffTrainingConfig = field(default_factory=FlowDiffTrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


if __name__ == "__main__":
    # Test config
    cfg = PipelineConfig()

    print("=== VAE Config (Texture-Optimized) ===")
    print(f"CLAHE enabled: {cfg.vae.clahe_config.enabled}")
    print(f"Morphological gradient: {cfg.vae.morph_grad_config.enabled}")
    print(f"\nLoss weights:")
    print(f"  λ_pix:        {cfg.vae.lambda_pix}")
    print(f"  λ_morph_grad: {cfg.vae.lambda_morph_grad} (HIGHEST!)")
    print(f"  λ_hf:         {cfg.vae.lambda_hf}")
    print(f"  λ_percep:     {cfg.vae.lambda_percep}")

    print("\n=== FlowDiff Config ===")
    print(f"Identity noise level: {cfg.flowdiff.identity_noise_level} (gentle)")
    print(f"Identity λ_global: {cfg.flowdiff.lambda_id_global}")

    print("\n=== Inference Config ===")
    print(f"Noise level: {cfg.inference.noise_level}")
    print(f"Anomaly weights: L1={cfg.inference.alpha_l1}, Grad={cfg.inference.beta_grad}, HF={cfg.inference.gamma_hf}")
