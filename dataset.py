import os
import numpy as np
import pydicom
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Optional, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class PanoramaXrayDataset(Dataset):
    """
    Panoramic X-ray DICOM dataset for VAE fine-tuning.
    - Reads .dcm files
    - Window level normalization to [-1, 1]
    - Grayscale → 3-channel replication
    - Returns: 256×256 images
    """

    def __init__(
            self,
            data_dir: str,
            file_list: Optional[List[Path]] = None,
            window_center: int = 2000,  # Typical for dental panoramic
            window_width: int = 4000,
            target_size: Tuple[int, int] = (256, 256),
            augment: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing .dcm files
            file_list: Optional list of specific files to use (for train/val split)
            window_center: WL (Window Level) for intensity normalization
            window_width: WW (Window Width)
            target_size: Output image size (H, W)
            augment: Whether to apply augmentation
        """
        self.data_dir = Path(data_dir)
        self.window_center = window_center
        self.window_width = window_width
        self.target_size = target_size
        self.augment = augment

        # Use provided file list or collect all .dcm files
        if file_list is not None:
            self.dicom_files = file_list
        else:
            self.dicom_files = sorted(list(self.data_dir.glob("**/*.dcm")))

        assert len(self.dicom_files) > 0, f"No DICOM files found"

        print(f"[Dataset] Loaded {len(self.dicom_files)} DICOM files")

        # Augmentation pipeline (weak, texture-preserving)
        # NOTE: Don't use ToTensorV2 here - we'll handle tensor conversion manually
        if self.augment:
            self.transform = A.Compose([
                A.Resize(*target_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=0,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*target_size),
            ])

    def __len__(self) -> int:
        return len(self.dicom_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            image: (3, H, W) float32 tensor, normalized to [-1, 1]
        """
        dcm_path = self.dicom_files[idx]

        # Read DICOM
        dcm = pydicom.dcmread(str(dcm_path))
        image = dcm.pixel_array.astype(np.float32)

        # Apply window level normalization
        image = self._apply_windowing(image)

        # Normalize to [-1, 1]
        image = self._normalize_to_range(image)

        # Apply augmentation & resize
        # Albumentations expects (H, W) for single channel
        transformed = self.transform(image=image)
        image = transformed["image"]  # Still (H, W) numpy array

        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).float()  # (H, W)
        image = image.unsqueeze(0)  # (1, H, W)

        # Grayscale → 3 channel replication
        image = image.repeat(3, 1, 1)  # (3, H, W)

        return image

    def _apply_windowing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply window level/width normalization.
        Formula: clip to [center - width/2, center + width/2]
        """
        lower = self.window_center - self.window_width / 2
        upper = self.window_center + self.window_width / 2

        image = np.clip(image, lower, upper)
        return image

    def _normalize_to_range(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize windowed image to [-1, 1]
        """
        # Current range: [center - width/2, center + width/2]
        # Map to [0, 1] first
        lower = self.window_center - self.window_width / 2
        upper = self.window_center + self.window_width / 2

        image = (image - lower) / (upper - lower)  # [0, 1]
        image = image * 2.0 - 1.0  # [-1, 1]

        return image.astype(np.float32)


def create_train_val_split(
        data_dir: str,
        val_ratio: float = 0.1,
        seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Create train/val split from a single directory.

    Args:
        data_dir: Directory containing all .dcm files
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        train_files: List of training file paths
        val_files: List of validation file paths
    """
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("**/*.dcm")))

    assert len(all_files) > 0, f"No DICOM files found in {data_dir}"

    # Split
    train_files, val_files = train_test_split(
        all_files,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    print(f"[Split] Total: {len(all_files)} files")
    print(f"[Split] Train: {len(train_files)} files")
    print(f"[Split] Val: {len(val_files)} files")

    return train_files, val_files


# Test code
if __name__ == "__main__":
    # Test split
    train_files, val_files = create_train_val_split(
        data_dir="/path/to/dicom/folder",
        val_ratio=0.1,
        seed=42,
    )

    # Create datasets
    train_dataset = PanoramaXrayDataset(
        data_dir="/path/to/dicom/folder",
        file_list=train_files,
        window_center=2000,
        window_width=4000,
        target_size=(256, 256),
        augment=True,
    )

    val_dataset = PanoramaXrayDataset(
        data_dir="/path/to/dicom/folder",
        file_list=val_files,
        window_center=2000,
        window_width=4000,
        target_size=(256, 256),
        augment=False,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Sample one
    img = train_dataset[0]
    print(f"Image shape: {img.shape}")  # (3, 256, 256)
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")  # ~[-1, 1]
