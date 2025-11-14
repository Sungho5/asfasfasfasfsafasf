"""
Dataset with Synthetic Lesions
정상 데이터에 synthetic lesion을 추가하여 self-supervised 학습
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("[Warning] pydicom not available. Install with: pip install pydicom")

from .lesion_synthesizer import DiverseLesionSynthesizer


class DSRNDataset(Dataset):
    """
    DSRN Dataset with Synthetic Lesions

    학습: 정상 X-ray에 가짜 병변을 추가
    목표: 가짜 병변 제거 → 원본 정상 복원
    """

    def __init__(self, data_root, image_size=256, train=True, lesion_prob=0.8,
                 window_center=2048, window_width=4096):
        """
        Args:
            data_root: Root directory containing normal X-ray images
            image_size: Target image size (default: 256)
            train: Training mode (True) or validation mode (False)
            lesion_prob: Probability of adding lesion
            window_center: Window level center for DICOM normalization (default: 2048)
            window_width: Window level width for DICOM normalization (default: 4096)
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.train = train
        self.lesion_prob = lesion_prob

        # Window level parameters for DICOM
        self.window_center = window_center
        self.window_width = window_width

        # Lesion synthesizer
        self.lesion_synthesizer = DiverseLesionSynthesizer()

        # Load image paths
        self.image_paths = self._load_image_paths()

        print(f"[DSRNDataset] Loaded {len(self.image_paths)} images from {data_root}")
        print(f"[DSRNDataset] Mode: {'Train' if train else 'Val'}")
        print(f"[DSRNDataset] Lesion probability: {lesion_prob}")
        print(f"[DSRNDataset] Window level: center={window_center}, width={window_width}")

    def _load_image_paths(self):
        """Load all image paths"""
        image_paths = []

        # Support multiple image formats (including DICOM)
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.dcm', '*.dicom']

        for ext in extensions:
            image_paths.extend(list(self.data_root.glob(f'**/{ext}')))

        # Sort for reproducibility
        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.data_root}")

        return image_paths

    def _apply_window_level(self, image, window_center, window_width):
        """
        Apply window level normalization to image

        Args:
            image: Raw pixel array
            window_center: Window level center (default: 2048)
            window_width: Window level width (default: 4096)

        Returns:
            Normalized image in [0, 1]
        """
        # Calculate min and max values
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        # Apply window level
        image = np.clip(image, img_min, img_max)

        # Normalize to [0, 1]
        image = (image - img_min) / (img_max - img_min)

        return image.astype(np.float32)

    def _load_and_preprocess(self, image_path):
        """
        Load and preprocess image (supports both standard images and DICOM)

        Returns:
            image: [H, W] numpy array, normalized to [0, 1]
        """
        image_path = Path(image_path)

        # Check if DICOM file
        if image_path.suffix.lower() in ['.dcm', '.dicom']:
            if not PYDICOM_AVAILABLE:
                raise ImportError("pydicom is required to load DICOM files. Install with: pip install pydicom")

            # Load DICOM
            dcm = pydicom.dcmread(str(image_path))
            image = dcm.pixel_array.astype(np.float32)

            # Apply window level normalization
            image = self._apply_window_level(image, self.window_center, self.window_width)

        else:
            # Load standard image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

        # Resize
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size))

        return image

    def _create_roi_mask(self, image):
        """
        Create ROI mask for lesion placement

        Simple version: Use entire image as ROI
        You can improve this by detecting actual tooth regions
        """
        roi_mask = np.ones_like(image, dtype=np.float32)

        # Avoid borders
        border = 30
        roi_mask[:border, :] = 0
        roi_mask[-border:, :] = 0
        roi_mask[:, :border] = 0
        roi_mask[:, -border:] = 0

        return roi_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            x_normal: [1, H, W] original normal image
            x_lesion: [1, H, W] image with synthetic lesion
            mask_lesion: [1, H, W] lesion mask
        """
        # Load normal image
        image_path = self.image_paths[idx]
        x_normal = self._load_and_preprocess(image_path)

        # Add synthetic lesion with probability
        if random.random() < self.lesion_prob:
            # Create ROI mask
            roi_mask = self._create_roi_mask(x_normal)

            # Synthesize lesion
            x_lesion, mask_lesion, _, _ = self.lesion_synthesizer.synthesize(
                x_normal.copy(),
                roi_mask,
                lesion_type='random'  # random, radiolucent, mixed
            )
        else:
            # No lesion (for prototype learning)
            x_lesion = x_normal.copy()
            mask_lesion = np.zeros_like(x_normal)

        # Convert to torch tensors
        x_normal = torch.from_numpy(x_normal).unsqueeze(0)  # [1, H, W]
        x_lesion = torch.from_numpy(x_lesion).unsqueeze(0)  # [1, H, W]
        mask_lesion = torch.from_numpy(mask_lesion).unsqueeze(0)  # [1, H, W]

        return {
            'x_normal': x_normal,  # Ground truth (original)
            'x_lesion': x_lesion,  # Input (with synthetic lesion)
            'mask_lesion': mask_lesion,  # Lesion location
        }


def create_dataloaders(config):
    """
    Create train and validation dataloaders

    Args:
        config: DSRN configuration

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split

    # Create dataset
    full_dataset = DSRNDataset(
        data_root=config.data_root,
        image_size=config.image_size,
        train=True,
        lesion_prob=config.lesion_prob,
        window_center=config.window_center,
        window_width=config.window_width
    )

    # Split train/val
    n_total = len(full_dataset)
    n_train = int(n_total * config.train_ratio)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"[Dataloaders] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    return train_loader, val_loader


if __name__ == "__main__":
    """Test"""
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        data_root: str = "/dataset/panorama_total"
        image_size: int = 256
        train_ratio: float = 0.8
        batch_size: int = 4
        num_workers: int = 0
        lesion_prob: float = 0.8

    # Test dataset
    dataset = DSRNDataset(
        data_root=TestConfig.data_root,
        image_size=TestConfig.image_size,
        lesion_prob=TestConfig.lesion_prob
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test one sample
    sample = dataset[0]

    print(f"\nSample keys: {sample.keys()}")
    print(f"x_normal shape: {sample['x_normal'].shape}")
    print(f"x_lesion shape: {sample['x_lesion'].shape}")
    print(f"mask_lesion shape: {sample['mask_lesion'].shape}")
    print(f"x_normal range: [{sample['x_normal'].min():.3f}, {sample['x_normal'].max():.3f}]")
    print(f"x_lesion range: [{sample['x_lesion'].min():.3f}, {sample['x_lesion'].max():.3f}]")

    print("\nDataset test passed!")
