"""
Phase 3: Flow + Diffusion Training
Train UNet with curriculum learning:
- Phase 1 (0-100k steps): Pure generative (L_diff + L_flow)
- Phase 2 (100k-300k steps): Add identity constraint (+ L_id)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Optional
import numpy as np

from latent_dataset import LatentDataset
from dataset import PanoramaXrayDataset, create_train_val_split
from flowdiff_model import FlowDiffUNet, EMAModel
from flowdiff_loss import FlowDiffLoss, NoiseSchedule
from model import VAEFineTuner
from config import get_config_phase3, DataConfig


class MixedDataset(Dataset):
    """
    Mixed dataset that returns both latents and images.
    For identity constraint training.
    """

    def __init__(self, latent_dataset: LatentDataset, image_dataset: PanoramaXrayDataset):
        assert len(latent_dataset) == len(image_dataset), \
            "Latent and image datasets must have same length"
        self.latent_dataset = latent_dataset
        self.image_dataset = image_dataset

    def __len__(self):
        return len(self.latent_dataset)

    def __getitem__(self, idx):
        latent = self.latent_dataset[idx]  # (4, 32, 32)
        image = self.image_dataset[idx]  # (3, 256, 256)
        return latent, image


class FlowDiffTrainer:
    """
    Trainer for Flow + Diffusion model with curriculum learning.
    """

    def __init__(
            self,
            model: nn.Module,
            ema_model: EMAModel,
            criterion: FlowDiffLoss,
            train_loader_latent: DataLoader,
            train_loader_mixed: Optional[DataLoader],
            val_loader: Optional[DataLoader],
            optimizer: torch.optim.Optimizer,
            device: str,
            cfg: any,
    ):
        self.model = model.to(device)
        self.ema = ema_model
        self.criterion = criterion
        self.train_loader_latent = train_loader_latent
        self.train_loader_mixed = train_loader_mixed
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

        self.global_step = 0
        self.best_val_loss = float('inf')

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_identity_weight(self) -> float:
        """
        Compute identity loss weight with warm-up.

        Returns:
            lambda_id_current: current weight for identity loss
        """
        if self.global_step < self.cfg.phase1_steps:
            # Phase 1: No identity
            return 0.0
        elif self.global_step < self.cfg.identity_warmup_start:
            # Early Phase 2: Still 0
            return 0.0
        elif self.global_step < self.cfg.identity_warmup_end:
            # Warm-up: linear ramp 0 -> lambda_id_global
            progress = (self.global_step - self.cfg.identity_warmup_start) / \
                       (self.cfg.identity_warmup_end - self.cfg.identity_warmup_start)
            return progress * self.cfg.lambda_id_global
        else:
            # Full identity weight
            return self.cfg.lambda_id_global

    def train_step(self, use_identity: bool = False):
        """Single training step"""
        self.model.train()

        if use_identity and self.train_loader_mixed is not None:
            # Use mixed dataset (latents + images)
            try:
                latents, images = next(self.mixed_iter)
            except (StopIteration, AttributeError):
                self.mixed_iter = iter(self.train_loader_mixed)
                latents, images = next(self.mixed_iter)

            latents = latents.to(self.device)
            images = images.to(self.device)

            # Sample subset for identity constraint (save computation)
            num_identity = int(latents.shape[0] * self.cfg.identity_sample_ratio)
            if num_identity > 0:
                images_identity = images[:num_identity]
            else:
                images_identity = None
                use_identity = False

        else:
            # Use latent-only dataset
            try:
                latents = next(self.latent_iter)
            except (StopIteration, AttributeError):
                self.latent_iter = iter(self.train_loader_latent)
                latents = next(self.latent_iter)

            latents = latents.to(self.device)
            images_identity = None

        # Compute loss
        loss, loss_dict = self.criterion(
            model=self.model,
            z_data=latents,
            x_images=images_identity,
            use_identity=use_identity,
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)

        self.optimizer.step()

        # Update EMA
        self.ema.update()

        return loss_dict

    def train(self):
        """Main training loop with curriculum"""
        print(f"[Training] Starting Flow+Diff training for {self.cfg.total_steps} steps")
        print(f"[Curriculum] Phase 1 (Generative): 0 - {self.cfg.phase1_steps}")
        print(f"[Curriculum] Phase 2 (+ Identity): {self.cfg.phase1_steps} - {self.cfg.total_steps}")
        print(f"[Curriculum] Identity warm-up: {self.cfg.identity_warmup_start} - {self.cfg.identity_warmup_end}")

        pbar = tqdm(total=self.cfg.total_steps, desc="Training")

        while self.global_step < self.cfg.total_steps:
            # Determine if we should use identity constraint
            lambda_id = self.get_identity_weight()
            use_identity = lambda_id > 0.0

            # Update lambda in criterion
            self.criterion.lambda_id_global = lambda_id

            # Training step
            loss_dict = self.train_step(use_identity=use_identity)

            # Update progress bar
            phase = "Gen" if not use_identity else f"Gen+Id({lambda_id:.3f})"
            pbar.set_postfix({
                'phase': phase,
                'loss': f"{loss_dict.get('loss_total', 0):.4f}",
                'diff': f"{loss_dict.get('loss_diff', 0):.4f}",
                'flow': f"{loss_dict.get('loss_flow', 0):.4f}",
            })
            pbar.update(1)

            # Logging
            if self.cfg.use_wandb and self.global_step % 10 == 0:
                log_dict = {
                    'train/loss_total': loss_dict.get('loss_total', 0),
                    'train/loss_gen': loss_dict.get('loss_gen', 0),
                    'train/loss_diff': loss_dict.get('loss_diff', 0),
                    'train/loss_flow': loss_dict.get('loss_flow', 0),
                    'train/lambda_id': lambda_id,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                }

                if use_identity:
                    log_dict.update({
                        'train/loss_id': loss_dict.get('loss_id', 0),
                        'train/loss_id_pix': loss_dict.get('loss_id_pix', 0),
                        'train/loss_id_grad': loss_dict.get('loss_id_grad', 0),
                        'train/loss_id_hf': loss_dict.get('loss_id_hf', 0),
                    })

                wandb.log(log_dict)

            # Validation
            if self.global_step % self.cfg.val_every == 0 and self.global_step > 0:
                self.validate()

            # Checkpointing
            if self.global_step % self.cfg.save_every == 0 and self.global_step > 0:
                self.save_checkpoint()

            self.global_step += 1

        pbar.close()
        print("[Training] Complete!")

    @torch.no_grad()
    def validate(self):
        """Validation: sample from model and visualize"""
        self.model.eval()

        print(f"\n[Validation] Step {self.global_step}")

        # Apply EMA weights for inference
        self.ema.apply_shadow()

        # Sample from noise using Flow matching
        num_samples = 4
        z_noise = torch.randn(num_samples, 4, 32, 32).to(self.device)

        # Simple Euler integration for flow
        num_steps = 50
        dt = 1.0 / num_steps
        z_t = z_noise.clone()

        for step in range(num_steps):
            t = torch.full((num_samples,), step / num_steps, device=self.device)
            _, v_pred = self.model(z_t, t, return_both_heads=True)
            z_t = z_t + v_pred * dt

        # Decode samples with VAE
        if hasattr(self.criterion.identity_loss, 'vae'):
            vae = self.criterion.identity_loss.vae
            samples = vae.decode(z_t)

            # Log to wandb
            if self.cfg.use_wandb:
                import torchvision
                samples = (samples + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                grid = torchvision.utils.make_grid(samples, nrow=4)
                wandb.log({
                    'val/samples': wandb.Image(grid),
                    'global_step': self.global_step,
                })

        # Restore original weights
        self.ema.restore()

        self.model.train()

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        # Save latest
        latest_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, latest_path)
        print(f"[Checkpoint] Saved at step {self.global_step}")

        # Also save as latest
        latest_link = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_link)

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)


def main():
    # Load config
    cfg = get_config_phase3()
    data_cfg = DataConfig()

    # Seed
    torch.manual_seed(data_cfg.seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}")

    # ===== Load Latent Dataset =====
    train_latent_path = Path(cfg.latent_dir) / "latents_train.npy"
    val_latent_path = Path(cfg.latent_dir) / "latents_val.npy"

    assert train_latent_path.exists(), f"Latent file not found: {train_latent_path}"
    assert val_latent_path.exists(), f"Latent file not found: {val_latent_path}"

    train_latent_dataset = LatentDataset(str(train_latent_path), normalize=False)
    val_latent_dataset = LatentDataset(str(val_latent_path), normalize=False)

    # ===== Load Image Dataset (for identity constraint) =====
    train_files, val_files = create_train_val_split(
        data_dir=data_cfg.data_dir,
        val_ratio=data_cfg.val_ratio,
        seed=data_cfg.seed,
    )

    train_image_dataset = PanoramaXrayDataset(
        data_dir=data_cfg.data_dir,
        file_list=train_files,
        window_center=data_cfg.window_center,
        window_width=data_cfg.window_width,
        target_size=data_cfg.target_size,
        augment=False,  # No augmentation for identity constraint
    )

    # ===== Mixed Dataset =====
    train_mixed_dataset = MixedDataset(train_latent_dataset, train_image_dataset)

    # ===== DataLoaders =====
    train_loader_latent = DataLoader(
        train_latent_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    train_loader_mixed = DataLoader(
        train_mixed_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_latent_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    # ===== Load VAE (for identity constraint) =====
    print(f"[Model] Loading VAE from {cfg.vae_checkpoint}")
    vae = VAEFineTuner(freeze_encoder=True)
    vae_ckpt = torch.load(cfg.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    print(f"[Model] VAE loaded from epoch {vae_ckpt.get('epoch', '?')}")

    # ===== Create Flow+Diff Model =====
    model = FlowDiffUNet(
        in_channels=cfg.unet_config.in_channels,
        out_channels=cfg.unet_config.out_channels,
        model_channels=cfg.unet_config.model_channels,
        num_res_blocks=cfg.unet_config.num_res_blocks,
        attention_resolutions=cfg.unet_config.attention_resolutions,
        channel_mult=cfg.unet_config.channel_mult,
        dropout=cfg.unet_config.dropout,
        num_heads=cfg.unet_config.num_heads,
        use_dual_head=cfg.unet_config.use_dual_head,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Flow+Diff UNet created with {total_params:,} parameters")

    # ===== EMA Model =====
    ema = EMAModel(model, decay=cfg.ema_decay)

    # ===== Noise Schedule =====
    noise_schedule = NoiseSchedule(
        num_timesteps=cfg.diffusion_config.num_timesteps,
        beta_start=cfg.diffusion_config.beta_start,
        beta_end=cfg.diffusion_config.beta_end,
        schedule_type=cfg.diffusion_config.schedule_type,
    )

    # ===== Loss Function =====
    criterion = FlowDiffLoss(
        noise_schedule=noise_schedule,
        vae_model=vae,
        lambda_diff=cfg.lambda_diff,
        lambda_flow=cfg.lambda_flow,
        lambda_id_global=cfg.lambda_id_global,
        identity_noise_level=cfg.identity_noise_level,
        lambda_id_pix=cfg.lambda_id_pix,
        lambda_id_grad=cfg.lambda_id_grad,
        lambda_id_hf=cfg.lambda_id_hf,
    )

    # ===== Optimizer =====
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ===== Wandb =====
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            config={
                'cfg': cfg.__dict__,
                'data_cfg': data_cfg.__dict__,
            }
        )

    # ===== Trainer =====
    trainer = FlowDiffTrainer(
        model=model,
        ema_model=ema,
        criterion=criterion,
        train_loader_latent=train_loader_latent,
        train_loader_mixed=train_loader_mixed,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
    )

    # ===== Train =====
    trainer.train()

    print("[Complete] Flow+Diff training finished!")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
