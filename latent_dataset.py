"""
Latent Dataset for Flow+Diff Training
Loads pre-extracted latent representations.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class LatentDataset(Dataset):
    """
    Dataset of latent representations (4, 32, 32).
    For Flow+Diff training on normal latent manifold.
    """

    def __init__(
            self,
            latent_file: str,
            normalize: bool = False,
    ):
        """
        Args:
            latent_file: Path to .npy file with latents (N, 4, 32, 32)
            normalize: If True, normalize latents to zero mean, unit std
        """
        self.latent_file = Path(latent_file)
        assert self.latent_file.exists(), f"Latent file not found: {latent_file}"

        # Load latents
        self.latents = np.load(self.latent_file)  # (N, 4, 32, 32)
        assert self.latents.ndim == 4 and self.latents.shape[1] == 4, \
            f"Expected shape (N, 4, H, W), got {self.latents.shape}"

        print(f"[LatentDataset] Loaded {len(self.latents)} latents from {latent_file}")
        print(f"[LatentDataset] Shape: {self.latents.shape}")
        print(f"[LatentDataset] Range: [{self.latents.min():.3f}, {self.latents.max():.3f}]")
        print(f"[LatentDataset] Mean: {self.latents.mean():.3f}, Std: {self.latents.std():.3f}")

        # Optional normalization
        if normalize:
            self.mean = self.latents.mean()
            self.std = self.latents.std()
            self.latents = (self.latents - self.mean) / (self.std + 1e-8)
            print(f"[LatentDataset] Normalized to mean=0, std=1")
        else:
            self.mean = None
            self.std = None

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            latent: (4, 32, 32) float32 tensor
        """
        latent = self.latents[idx].astype(np.float32)
        return torch.from_numpy(latent)


# Test
if __name__ == "__main__":
    # Create dummy latents for testing
    dummy_latents = np.random.randn(100, 4, 32, 32).astype(np.float32)
    dummy_path = Path("./test_latents.npy")
    np.save(dummy_path, dummy_latents)

    # Load dataset
    dataset = LatentDataset(str(dummy_path), normalize=False)

    print(f"\nDataset length: {len(dataset)}")

    # Sample
    latent = dataset[0]
    print(f"Sample shape: {latent.shape}")
    print(f"Sample dtype: {latent.dtype}")

    # Clean up
    dummy_path.unlink()
