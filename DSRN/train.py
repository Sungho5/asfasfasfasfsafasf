"""
DSRN Training Script

Phase 1: Normal Prototype Learning
Phase 2: Self-Supervised Reconstruction Training
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from config import DSRNConfig
from models.dsrn import DSRN
from data.dataset import create_dataloaders
from utils.visualization import visualize_results


class DSRNTrainer:
    """DSRN Trainer"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # Create model
        self.model = DSRN(config).to(self.device)

        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(config)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Best metrics
        self.best_val_loss = float('inf')

        print("[Trainer] Initialized")

    def compute_losses(self, x_normal, x_lesion, mask_lesion, x_recon, anomaly_map):
        """
        Compute all losses

        Args:
            x_normal: [B, 1, H, W] ground truth (original)
            x_lesion: [B, 1, H, W] input (with synthetic lesion)
            mask_lesion: [B, 1, H, W] lesion mask
            x_recon: [B, 1, H, W] reconstructed output
            anomaly_map: [B, 1, H, W] predicted anomaly map

        Returns:
            loss_dict: Dictionary of losses
        """
        # 1. Reconstruction Loss
        #    재구성 결과가 원본 정상과 같아야 함

        # 1-1. Global reconstruction
        loss_recon_global = F.mse_loss(x_recon, x_normal)

        # 1-2. 🔥 Lesion region reconstruction (병변 부분을 정확히 복구했는지)
        if mask_lesion.sum() > 0:
            loss_recon_lesion = F.mse_loss(
                x_recon * mask_lesion,
                x_normal * mask_lesion
            )
        else:
            loss_recon_lesion = torch.tensor(0.0, device=x_recon.device)

        # Combined
        loss_recon = loss_recon_global + 2.0 * loss_recon_lesion  # 병변 부분에 2배 가중치

        # 2. Anomaly Detection Loss (Enhanced!)
        #    병변 위치를 정확히 찾아야 함

        # 2-1. BCE Loss (pixel-wise)
        loss_anomaly_bce = F.binary_cross_entropy(anomaly_map, mask_lesion)

        # 2-2. 🔥 Dice Loss (region-wise, 경계를 더 정확하게)
        smooth = 1e-5
        intersection = (anomaly_map * mask_lesion).sum(dim=[2, 3])
        union = anomaly_map.sum(dim=[2, 3]) + mask_lesion.sum(dim=[2, 3])
        dice = (2. * intersection + smooth) / (union + smooth)
        loss_anomaly_dice = 1 - dice.mean()

        # 2-3. 🔥 Focal Loss (hard example에 집중)
        alpha = 0.25
        gamma = 2.0
        bce_loss = F.binary_cross_entropy(anomaly_map, mask_lesion, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = p if y=1, else 1-p
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        loss_anomaly_focal = focal_loss.mean()

        # Combined anomaly loss
        loss_anomaly = loss_anomaly_bce + loss_anomaly_dice + 0.5 * loss_anomaly_focal

        # 3. Identity Preservation Loss
        #    정상 부분(mask 밖)은 그대로 유지
        mask_normal = 1 - mask_lesion
        loss_identity = F.mse_loss(
            x_recon * mask_normal,
            x_lesion * mask_normal
        )

        # 4. Perceptual Loss (optional - L1 in gradient domain)
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

        # Compute gradients
        grad_recon_x = F.conv2d(x_recon, sobel_x, padding=1)
        grad_recon_y = F.conv2d(x_recon, sobel_y, padding=1)
        grad_normal_x = F.conv2d(x_normal, sobel_x, padding=1)
        grad_normal_y = F.conv2d(x_normal, sobel_y, padding=1)

        loss_perceptual = F.l1_loss(grad_recon_x, grad_normal_x) + F.l1_loss(grad_recon_y, grad_normal_y)

        # Total loss
        total_loss = (
            self.config.lambda_recon * loss_recon +
            self.config.lambda_anomaly * loss_anomaly +
            self.config.lambda_identity * loss_identity +
            self.config.lambda_perceptual * loss_perceptual
        )

        return {
            'total': total_loss,
            'recon': loss_recon,
            'anomaly': loss_anomaly,
            'identity': loss_identity,
            'perceptual': loss_perceptual,
            # Detailed losses for monitoring
            'recon_global': loss_recon_global,
            'recon_lesion': loss_recon_lesion,
            'anomaly_bce': loss_anomaly_bce,
            'anomaly_dice': loss_anomaly_dice,
            'anomaly_focal': loss_anomaly_focal,
        }

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()

        total_losses = {
            'total': 0,
            'recon': 0,
            'anomaly': 0,
            'identity': 0,
            'perceptual': 0,
            'recon_global': 0,
            'recon_lesion': 0,
            'anomaly_bce': 0,
            'anomaly_dice': 0,
            'anomaly_focal': 0,
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            x_normal = batch['x_normal'].to(self.device)  # [B, 1, H, W]
            x_lesion = batch['x_lesion'].to(self.device)  # [B, 1, H, W]
            mask_lesion = batch['mask_lesion'].to(self.device)  # [B, 1, H, W]

            # Phase 1: Update prototypes with normal images (no lesion)
            # Only update when mask is all zeros
            has_no_lesion = (mask_lesion.sum(dim=[1, 2, 3]) == 0)
            if has_no_lesion.any():
                self.model.update_prototypes(x_normal[has_no_lesion])

            # Phase 2: Reconstruction training
            # Forward
            x_recon, anomaly_map, fusion_weights = self.model(x_lesion)

            # Compute losses
            losses = self.compute_losses(
                x_normal, x_lesion, mask_lesion,
                x_recon, anomaly_map
            )

            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            self.optimizer.step()

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon'].item():.4f}",
                'anom': f"{losses['anomaly'].item():.4f}"
            })

        # Average losses
        for key in total_losses:
            total_losses[key] /= len(self.train_loader)

        return total_losses

    @torch.no_grad()
    def validate(self, epoch):
        """Validate"""
        self.model.eval()

        total_losses = {
            'total': 0,
            'recon': 0,
            'anomaly': 0,
            'identity': 0,
            'perceptual': 0,
            'recon_global': 0,
            'recon_lesion': 0,
            'anomaly_bce': 0,
            'anomaly_dice': 0,
            'anomaly_focal': 0,
        }

        pbar = tqdm(self.val_loader, desc='Validation')

        for batch in pbar:
            x_normal = batch['x_normal'].to(self.device)
            x_lesion = batch['x_lesion'].to(self.device)
            mask_lesion = batch['mask_lesion'].to(self.device)

            # Forward
            x_recon, anomaly_map, fusion_weights = self.model(x_lesion)

            # Compute losses
            losses = self.compute_losses(
                x_normal, x_lesion, mask_lesion,
                x_recon, anomaly_map
            )

            # Accumulate
            for key in total_losses:
                total_losses[key] += losses[key].item()

        # Average
        for key in total_losses:
            total_losses[key] /= len(self.val_loader)

        return total_losses

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Saved best model at epoch {epoch}")

        # Save periodic
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"[Checkpoint] Loaded from {checkpoint_path}")

        return checkpoint['epoch']

    def train(self):
        """Main training loop"""
        print("=" * 70)
        print("Starting DSRN Training")
        print("=" * 70)

        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.validate(epoch)

            # Print
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"Train Loss: {train_losses['total']:.4f} | "
                  f"Recon: {train_losses['recon']:.4f} | "
                  f"Anomaly: {train_losses['anomaly']:.4f} | "
                  f"Identity: {train_losses['identity']:.4f}")
            print(f"  └─ Recon: Global={train_losses['recon_global']:.4f} Lesion={train_losses['recon_lesion']:.4f}")
            print(f"  └─ Anomaly: BCE={train_losses['anomaly_bce']:.4f} Dice={train_losses['anomaly_dice']:.4f} Focal={train_losses['anomaly_focal']:.4f}")
            print(f"Val Loss:   {val_losses['total']:.4f} | "
                  f"Recon: {val_losses['recon']:.4f} | "
                  f"Anomaly: {val_losses['anomaly']:.4f} | "
                  f"Identity: {val_losses['identity']:.4f}")
            print(f"  └─ Dice Score: {1 - val_losses['anomaly_dice']:.4f}")  # Higher is better

            # Tensorboard
            for key in train_losses:
                self.writer.add_scalar(f'train/{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'val/{key}', val_losses[key], epoch)

            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Visualize
            if epoch % self.config.vis_freq == 0:
                self.visualize_epoch(epoch)

            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']

            self.save_checkpoint(epoch, is_best=is_best)

            # Step scheduler
            self.scheduler.step()

        print("=" * 70)
        print("Training Complete!")
        print("=" * 70)

    @torch.no_grad()
    def visualize_epoch(self, epoch):
        """Visualize results"""
        self.model.eval()

        # Get one batch
        batch = next(iter(self.val_loader))
        x_normal = batch['x_normal'].to(self.device)
        x_lesion = batch['x_lesion'].to(self.device)
        mask_lesion = batch['mask_lesion'].to(self.device)

        # Forward
        x_recon, anomaly_map, fusion_weights = self.model(x_lesion)

        # Visualize
        save_path = Path(self.config.output_dir) / f'epoch_{epoch:03d}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        visualize_results(
            x_normal=x_normal[:4],
            x_lesion=x_lesion[:4],
            x_recon=x_recon[:4],
            mask_lesion=mask_lesion[:4],
            anomaly_map=anomaly_map[:4],
            fusion_weights=fusion_weights[:4],
            save_path=str(save_path)
        )


def main():
    """Main"""
    # Load config
    config = DSRNConfig()

    # Create trainer
    trainer = DSRNTrainer(config)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
