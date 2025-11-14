"""
Configuration for VAE + Flow + Diffusion pipeline for dental X-ray anomaly detection.
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
class VAEConfig:
    """VAE fine-tuning configuration (Phase 1)"""
    pretrained_model: str = 'stabilityai/sd-vae-ft-mse'
    freeze_encoder: bool = True
    unfreeze_last_n_blocks: int = 0  # 0 = fully frozen, 1+ = unfreeze last N blocks

    # Loss weights (based on your recommendations)
    lambda_pix: float = 1.0      # Pixel reconstruction (L1) - baseline
    lambda_grad: float = 0.5     # Gradient loss
    lambda_hf: float = 0.3       # High-frequency (Laplacian)
    lambda_ssim: float = 0.1     # SSIM
    beta_kl: float = 1e-6        # KL divergence (keep latent distribution stable)

    # Training
    num_epochs: int = 50
    lr: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Checkpointing
    output_dir: str = './outputs/phase1_vae'
    save_every: int = 5  # Save checkpoint every N epochs

    # Wandb
    use_wandb: bool = False
    wandb_project: str = 'dental-xray-vae'


@dataclass
class LatentConfig:
    """Latent dataset generation configuration (Phase 2)"""
    vae_checkpoint: str = './outputs/phase1_vae/checkpoint_best.pt'
    output_dir: str = './outputs/phase2_latents'
    batch_size: int = 16  # Larger batch for faster extraction
    use_mean: bool = True  # Use posterior mean (not sampling)


@dataclass
class DiffusionConfig:
    """Diffusion model configuration"""
    # Noise schedule
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: str = 'linear'  # 'linear' or 'cosine'

    # Denoising
    prediction_type: str = 'epsilon'  # 'epsilon' or 'v_prediction'


@dataclass
class FlowConfig:
    """Flow matching configuration"""
    # Time range for flow matching
    t_min: float = 0.0
    t_max: float = 1.0

    # Base distribution
    base_std: float = 1.0  # N(0, I) noise


@dataclass
class UNetConfig:
    """UNet backbone configuration for Flow+Diff"""
    in_channels: int = 4  # Latent channels
    out_channels: int = 4  # Same as input
    model_channels: int = 128  # Base channel count
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (8, 16)  # Apply attention at these resolutions
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)  # Channel multipliers per resolution
    dropout: float = 0.0
    use_checkpoint: bool = False  # Gradient checkpointing for memory
    num_heads: int = 4

    # Dual head
    use_dual_head: bool = True  # Diffusion + Flow heads


@dataclass
class FlowDiffTrainingConfig:
    """Flow+Diff training configuration (Phase 3)"""

    # Data
    latent_dir: str = './outputs/phase2_latents'
    vae_checkpoint: str = './outputs/phase1_vae/checkpoint_best.pt'

    # Model
    unet_config: UNetConfig = field(default_factory=UNetConfig)
    diffusion_config: DiffusionConfig = field(default_factory=DiffusionConfig)
    flow_config: FlowConfig = field(default_factory=FlowConfig)

    # Curriculum learning phases
    phase1_steps: int = 100_000  # Pure generative (L_diff + L_flow)
    phase2_steps: int = 200_000  # Add identity constraint
    total_steps: int = 300_000

    # Loss weights - Phase 1 (Generative only)
    lambda_diff: float = 1.0
    lambda_flow: float = 0.5

    # Loss weights - Phase 2 (Add identity)
    lambda_id_global: float = 0.1  # Overall identity weight
    lambda_id_pix: float = 1.0     # Pixel component
    lambda_id_grad: float = 0.5    # Gradient component
    lambda_id_hf: float = 0.3      # High-frequency component

    # Identity constraint settings
    identity_noise_level: float = 0.4  # t0 for adding noise before denoising
    identity_sample_ratio: float = 0.2  # Fraction of batch to use for identity

    # Warm-up for identity constraint
    identity_warmup_start: int = 90_000   # Start warming up identity loss
    identity_warmup_end: int = 110_000    # Finish warm-up

    # Training
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    ema_decay: float = 0.9999  # Exponential moving average for inference

    # Checkpointing
    output_dir: str = './outputs/phase3_flowdiff'
    save_every: int = 10_000  # Save every N steps
    val_every: int = 5_000    # Validate every N steps

    # Wandb
    use_wandb: bool = False
    wandb_project: str = 'dental-xray-flowdiff'


@dataclass
class InferenceConfig:
    """Inference configuration (Phase 4)"""
    vae_checkpoint: str = './outputs/phase1_vae/checkpoint_best.pt'
    flowdiff_checkpoint: str = './outputs/phase3_flowdiff/checkpoint_best.pt'

    # Denoising settings
    noise_level: float = 0.4  # t0 for normalizer
    num_denoising_steps: int = 1  # Single-step vs multi-step
    use_ddim: bool = False  # Use DDIM sampler

    # Anomaly map
    alpha_l1: float = 1.0    # Weight for L1 difference
    beta_grad: float = 0.5   # Weight for gradient difference
    gamma_hf: float = 0.3    # Weight for HF difference

    # Output
    output_dir: str = './outputs/phase4_inference'
    save_heatmaps: bool = True


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


# Complete pipeline config
@dataclass
class PipelineConfig:
    """Full pipeline configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    flowdiff: FlowDiffTrainingConfig = field(default_factory=FlowDiffTrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


if __name__ == "__main__":
    # Test config creation
    config = PipelineConfig()

    print("=== Data Config ===")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Image size: {config.data.target_size}")

    print("\n=== VAE Config ===")
    print(f"Freeze encoder: {config.vae.freeze_encoder}")
    print(f"Loss weights: pix={config.vae.lambda_pix}, grad={config.vae.lambda_grad}, hf={config.vae.lambda_hf}")
    print(f"KL weight: {config.vae.beta_kl}")

    print("\n=== Flow+Diff Config ===")
    print(f"Phase 1 steps: {config.flowdiff.phase1_steps}")
    print(f"Phase 2 steps: {config.flowdiff.phase2_steps}")
    print(f"Identity warm-up: {config.flowdiff.identity_warmup_start} -> {config.flowdiff.identity_warmup_end}")
    print(f"Loss weights: diff={config.flowdiff.lambda_diff}, flow={config.flowdiff.lambda_flow}")

    print("\n=== Inference Config ===")
    print(f"Noise level: {config.inference.noise_level}")
    print(f"Anomaly map weights: L1={config.inference.alpha_l1}, Grad={config.inference.beta_grad}, HF={config.inference.gamma_hf}")
