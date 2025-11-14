import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Optional

from dataset import PanoramaXrayDataset, create_train_val_split
from model import VAEFineTuner, VAELoss


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
            use_wandb: bool = True,
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
            'percep': 0.0,  # NEW
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # 1.0 → 0.5

            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]

            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total'],
                'pix': loss_dict['pix'],
                'percep': loss_dict['percep'],  # NEW
            })

            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss_dict['total'],
                    'train/loss_pix': loss_dict['pix'],
                    'train/loss_grad': loss_dict['grad'],
                    'train/loss_hf': loss_dict['hf'],
                    'train/loss_ssim': loss_dict['ssim'],
                    'train/loss_percep': loss_dict['percep'],  # NEW
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
            'percep': 0.0,  # NEW
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
                'val/loss_percep': val_losses['percep'],  # NEW
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
                  f"(pix:{train_losses['pix']:.4f}, percep:{train_losses['percep']:.4f})")

            # Validate
            val_losses = self.validate(epoch)
            print(f"[Epoch {epoch}] Val Loss: {val_losses['total']:.4f} "
                  f"(pix:{val_losses['pix']:.4f}, percep:{val_losses['percep']:.4f})\n")

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']

            self.save_checkpoint(epoch, val_losses['total'], is_best=is_best)


def main():
    # Config
    config = {
        'data_dir': '/home/imgadmin/DATA1/sungho/synology/sungho/NEW_Anomaly/data/2048/',
        'val_ratio': 0.1,
        'output_dir': './outputs/vae_finetuning_percep_encoder',  # NEW folder
        'pretrained_model': 'stabilityai/sd-vae-ft-mse',

        # Encoder settings
        'freeze_encoder': True,
        'unfreeze_last_n_blocks': 1,  # NEW: Unfreeze last 1 block

        # Resume from perceptual loss checkpoint
        'resume_from': './outputs/vae_finetuning_percep/checkpoint_best.pt',

        # Dataset
        'window_center': 2000,
        'window_width': 4000,
        'target_size': (256, 256),

        # Training - MORE CONSERVATIVE
        'batch_size': 8,
        'num_epochs': 30,      # 30 epoch
        'lr': 2e-5,            # LOWER LR (was 5e-5)
        'weight_decay': 1e-5,
        'num_workers': 4,

        # Loss weights - SAME AS BEFORE
        'lambda_pix': 0.5,
        'lambda_grad': 1.0,
        'lambda_hf': 0.8,
        'lambda_ssim': 0.3,
        'lambda_percep': 1.0,
        'beta_kl': 5e-7,       # LOWER KL weight (was 1e-6)

        # Misc
        'use_wandb': False,
        'wandb_project': 'vae-dental-xray-percep-encoder',
        'seed': 42,
    }

    # Seed
    torch.manual_seed(config['seed'])

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}")

    # Create train/val split
    train_files, val_files = create_train_val_split(
        data_dir=config['data_dir'],
        val_ratio=config['val_ratio'],
        seed=config['seed'],
    )

    # Datasets
    train_dataset = PanoramaXrayDataset(
        data_dir=config['data_dir'],
        file_list=train_files,
        window_center=config['window_center'],
        window_width=config['window_width'],
        target_size=config['target_size'],
        augment=True,
    )

    val_dataset = PanoramaXrayDataset(
        data_dir=config['data_dir'],
        file_list=val_files,
        window_center=config['window_center'],
        window_width=config['window_width'],
        target_size=config['target_size'],
        augment=False,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    # Model
    model = VAEFineTuner(
        pretrained_model=config['pretrained_model'],
        freeze_encoder=config['freeze_encoder'],
        unfreeze_last_n_blocks=config.get('unfreeze_last_n_blocks', 0),  # NEW
    )

    # Loss (with perceptual loss)
    criterion = VAELoss(
        lambda_pix=config['lambda_pix'],
        lambda_grad=config['lambda_grad'],
        lambda_hf=config['lambda_hf'],
        lambda_ssim=config['lambda_ssim'],
        lambda_percep=config['lambda_percep'],
        beta_kl=config['beta_kl'],
    )

    encoder_params = []
    decoder_params = []

    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

    # Create optimizer with conditional parameter groups
    param_groups = []

    if len(encoder_params) > 0:
        param_groups.append({'params': encoder_params, 'lr': config['lr'] * 0.5})
        print(f"[Optimizer] Encoder params: {len(encoder_params)}, LR: {config['lr'] * 0.5}")
    else:
        print(f"[Optimizer] No encoder params to train")

    if len(decoder_params) > 0:
        param_groups.append({'params': decoder_params, 'lr': config['lr']})
        print(f"[Optimizer] Decoder params: {len(decoder_params)}, LR: {config['lr']}")
    else:
        print(f"[Optimizer] No decoder params to train")

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found!")

    optimizer = AdamW(param_groups, weight_decay=config['weight_decay'])

    print(f"[Optimizer] Encoder params: {len(encoder_params)}, LR: {config['lr'] * 0.5}")
    print(f"[Optimizer] Decoder params: {len(decoder_params)}, LR: {config['lr']}")

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-7,  # Lower min LR
    )

    # Resume from checkpoint
    if config.get('resume_from') and Path(config['resume_from']).exists():
        print(f"[Resume] Loading from {config['resume_from']}")
        checkpoint = torch.load(config['resume_from'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Don't load optimizer - fresh start with new param groups
        print(f"[Resume] Loaded model weights from epoch {checkpoint.get('epoch', 0)}")

    # Wandb
    if config['use_wandb']:
        wandb.init(
            project=config['wandb_project'],
            config=config,
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
        output_dir=config['output_dir'],
        use_wandb=config['use_wandb'],
    )

    # Train
    trainer.train(num_epochs=config['num_epochs'])

    print("[Training] Complete!")

    if config['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main()
