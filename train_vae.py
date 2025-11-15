"""
Phase 1: VAE Fine-tuning
Train decoder (and optionally encoder) on normal dental X-rays.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Optional
import torch.nn as nn

from dataset import PanoramaXrayDataset, create_train_val_split
from model import VAEFineTuner, VAELoss
from config import get_config_phase1, DataConfig


class VAETrainer:
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
            device: str,
            output_dir: str,
            use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int):
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        epoch_losses = {
            'total': 0.0,
            'pix': 0.0,
            'grad': 0.0,
            'hf': 0.0,
            'ssim': 0.0,
            'kl': 0.0,
        }

        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)

            # Forward
            x_rec, posterior = self.model(images)

            # Compute loss
            loss, loss_dict = self.criterion(
                images, x_rec,
                posterior['mean'],
                posterior['logvar']
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'pix': f"{loss_dict['pix']:.4f}",
                'kl': f"{loss_dict['kl']:.6f}",
            })

            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss_dict['total'],
                    'train/loss_pix': loss_dict['pix'],
                    'train/loss_grad': loss_dict['grad'],
                    'train/loss_hf': loss_dict['hf'],
                    'train/loss_ssim': loss_dict['ssim'],
                    'train/loss_kl': loss_dict['kl'],
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                })

            self.global_step += 1

        # Average epoch losses
        num_batches = len(self.train_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()

        val_losses = {
            'total': 0.0,
            'pix': 0.0,
            'grad': 0.0,
            'hf': 0.0,
            'ssim': 0.0,
            'kl': 0.0,
        }

        for images in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)

            # Forward
            x_rec, posterior = self.model(images)

            # Compute loss
            loss, loss_dict = self.criterion(
                images, x_rec,
                posterior['mean'],
                posterior['logvar']
            )

            # Accumulate
            for key in val_losses.keys():
                val_losses[key] += loss_dict[key]

        # Average
        num_batches = len(self.val_loader)
        for key in val_losses.keys():
            val_losses[key] /= num_batches

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/loss': val_losses['total'],
                'val/loss_pix': val_losses['pix'],
                'val/loss_grad': val_losses['grad'],
                'val/loss_hf': val_losses['hf'],
                'val/loss_ssim': val_losses['ssim'],
                'val/loss_kl': val_losses['kl'],
                'epoch': epoch,
            })

            # Log sample reconstructions
            self._log_reconstructions(images[:4], x_rec[:4], epoch)

        return val_losses

    def _log_reconstructions(self, images, reconstructions, epoch):
        """Log sample image reconstructions to wandb"""
        import torchvision

        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        reconstructions = (reconstructions + 1.0) / 2.0

        # Make grid
        grid_orig = torchvision.utils.make_grid(images, nrow=4)
        grid_rec = torchvision.utils.make_grid(reconstructions, nrow=4)

        wandb.log({
            'val/original': wandb.Image(grid_orig),
            'val/reconstructed': wandb.Image(grid_rec),
            'epoch': epoch,
        })

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Saved best model at epoch {epoch} with val_loss={val_loss:.4f}")

    def train(self, num_epochs: int):
        print(f"[Training] Starting for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            print(f"\n[Epoch {epoch}] Train Loss: {train_losses['total']:.4f} "
                  f"(pix:{train_losses['pix']:.4f}, kl:{train_losses['kl']:.6f})")

            # Validate
            val_losses = self.validate(epoch)
            print(f"[Epoch {epoch}] Val Loss: {val_losses['total']:.4f} "
                  f"(pix:{val_losses['pix']:.4f}, kl:{val_losses['kl']:.6f})\n")

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']

            self.save_checkpoint(epoch, val_losses['total'], is_best=is_best)


def main():
    # Load configs
    vae_cfg = get_config_phase1()
    data_cfg = DataConfig()

    # Seed
    torch.manual_seed(data_cfg.seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}")

    # Create train/val split
    train_files, val_files = create_train_val_split(
        data_dir=data_cfg.data_dir,
        val_ratio=data_cfg.val_ratio,
        seed=data_cfg.seed,
    )

    # Datasets
    train_dataset = PanoramaXrayDataset(
        data_dir=data_cfg.data_dir,
        file_list=train_files,
        window_center=data_cfg.window_center,
        window_width=data_cfg.window_width,
        target_size=data_cfg.target_size,
        augment=True,
    )

    val_dataset = PanoramaXrayDataset(
        data_dir=data_cfg.data_dir,
        file_list=val_files,
        window_center=data_cfg.window_center,
        window_width=data_cfg.window_width,
        target_size=data_cfg.target_size,
        augment=False,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    # Model
    model = VAEFineTuner(
        pretrained_model=vae_cfg.pretrained_model,
        freeze_encoder=vae_cfg.freeze_encoder,
        unfreeze_last_n_blocks=vae_cfg.unfreeze_last_n_blocks,
    )

    # Loss (using config weights)
    criterion = VAELoss(
        lambda_pix=vae_cfg.lambda_pix,
        lambda_grad=vae_cfg.lambda_grad,
        lambda_hf=vae_cfg.lambda_hf,
        lambda_ssim=vae_cfg.lambda_ssim,
        lambda_percep=0.0,  # Not using perceptual loss
        beta_kl=vae_cfg.beta_kl,
    )

    # Print loss weights
    print("\n[Loss Weights]")
    print(f"  λ_pix:  {vae_cfg.lambda_pix}")
    print(f"  λ_grad: {vae_cfg.lambda_grad}")
    print(f"  λ_hf:   {vae_cfg.lambda_hf}")
    print(f"  λ_ssim: {vae_cfg.lambda_ssim}")
    print(f"  β_kl:   {vae_cfg.beta_kl}")
    print()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=vae_cfg.lr,
        weight_decay=vae_cfg.weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=vae_cfg.num_epochs,
        eta_min=1e-7,
    )

    # Wandb
    if vae_cfg.use_wandb:
        wandb.init(
            project=vae_cfg.wandb_project,
            config={
                'vae': vae_cfg.__dict__,
                'data': data_cfg.__dict__,
            }
        )

    # Trainer
    trainer = VAETrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=vae_cfg.output_dir,
        use_wandb=vae_cfg.use_wandb,
    )

    # Train
    trainer.train(num_epochs=vae_cfg.num_epochs)

    print("[Training] Complete!")

    if vae_cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
