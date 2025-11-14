"""
Phase 2: Latent Dataset Generation
Extract latent representations from normal X-rays using fine-tuned VAE.
This creates the "normal latent manifold" for Flow+Diff training.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle

from dataset import PanoramaXrayDataset, create_train_val_split
from model import VAEFineTuner
from config import get_config_phase2, DataConfig


class LatentExtractor:
    """Extract and save latent representations from VAE encoder"""

    def __init__(
            self,
            vae_model: VAEFineTuner,
            device: str,
            output_dir: str,
            use_mean: bool = True,
    ):
        """
        Args:
            vae_model: Fine-tuned VAE
            device: cuda or cpu
            output_dir: Where to save latents
            use_mean: If True, use posterior mean. If False, sample.
        """
        self.model = vae_model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mean = use_mean

    @torch.no_grad()
    def extract_from_loader(
            self,
            dataloader: DataLoader,
            split_name: str,
    ) -> dict:
        """
        Extract latents from a dataloader.

        Args:
            dataloader: DataLoader for X-ray images
            split_name: 'train' or 'val'

        Returns:
            latent_dict: {
                'latents': np.ndarray of shape (N, 4, 32, 32),
                'file_paths': list of source file paths,
            }
        """
        latents = []
        file_indices = []

        print(f"[Extract] Processing {split_name} set...")

        for batch_idx, images in enumerate(tqdm(dataloader, desc=f"Extracting {split_name}")):
            images = images.to(self.device)  # (B, 3, 256, 256)

            # Encode to latent
            z = self.model.encode(images, sample=not self.use_mean)  # (B, 4, 32, 32)

            # Move to CPU and store
            latents.append(z.cpu().numpy())

            # Track batch indices
            batch_size = images.shape[0]
            start_idx = batch_idx * dataloader.batch_size
            file_indices.extend(range(start_idx, start_idx + batch_size))

        # Concatenate all batches
        latents = np.concatenate(latents, axis=0)  # (N, 4, 32, 32)

        print(f"[Extract] {split_name}: Extracted {latents.shape[0]} latents")
        print(f"[Extract] Latent shape: {latents.shape}")
        print(f"[Extract] Latent range: [{latents.min():.3f}, {latents.max():.3f}]")
        print(f"[Extract] Latent mean: {latents.mean():.3f}, std: {latents.std():.3f}")

        # Get file paths
        dataset = dataloader.dataset
        file_paths = [str(dataset.dicom_files[i]) for i in file_indices]

        latent_dict = {
            'latents': latents,
            'file_paths': file_paths,
        }

        return latent_dict

    def save_latents(self, latent_dict: dict, split_name: str):
        """Save latents to disk"""
        save_path = self.output_dir / f"latents_{split_name}.pkl"

        with open(save_path, 'wb') as f:
            pickle.dump(latent_dict, f)

        print(f"[Save] Saved {split_name} latents to {save_path}")

        # Also save as individual .npy for easier loading during training
        latents_only_path = self.output_dir / f"latents_{split_name}.npy"
        np.save(latents_only_path, latent_dict['latents'])
        print(f"[Save] Saved latents array to {latents_only_path}")


def main():
    # Load configs
    latent_cfg = get_config_phase2()
    data_cfg = DataConfig()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}")

    # Load fine-tuned VAE
    print(f"[Model] Loading VAE from {latent_cfg.vae_checkpoint}")
    vae = VAEFineTuner(freeze_encoder=True)

    checkpoint = torch.load(latent_cfg.vae_checkpoint, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Model] Loaded VAE from epoch {checkpoint.get('epoch', '?')}")

    # Create train/val split (same as Phase 1)
    train_files, val_files = create_train_val_split(
        data_dir=data_cfg.data_dir,
        val_ratio=data_cfg.val_ratio,
        seed=data_cfg.seed,
    )

    # Datasets (NO augmentation for latent extraction)
    train_dataset = PanoramaXrayDataset(
        data_dir=data_cfg.data_dir,
        file_list=train_files,
        window_center=data_cfg.window_center,
        window_width=data_cfg.window_width,
        target_size=data_cfg.target_size,
        augment=False,  # Important: no augmentation
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
        batch_size=latent_cfg.batch_size,
        shuffle=False,  # Keep order for file path mapping
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=latent_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    # Extractor
    extractor = LatentExtractor(
        vae_model=vae,
        device=device,
        output_dir=latent_cfg.output_dir,
        use_mean=latent_cfg.use_mean,
    )

    # Extract train latents
    train_latents = extractor.extract_from_loader(train_loader, split_name='train')
    extractor.save_latents(train_latents, split_name='train')

    # Extract val latents
    val_latents = extractor.extract_from_loader(val_loader, split_name='val')
    extractor.save_latents(val_latents, split_name='val')

    print("\n[Complete] Latent extraction finished!")
    print(f"[Output] Latents saved to {latent_cfg.output_dir}")
    print(f"  - train: {train_latents['latents'].shape[0]} samples")
    print(f"  - val:   {val_latents['latents'].shape[0]} samples")


if __name__ == "__main__":
    main()
