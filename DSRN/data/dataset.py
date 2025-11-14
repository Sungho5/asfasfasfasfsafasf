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
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        return image

    def _create_roi_mask(self, image):
        """
        Create ROI mask for lesion placement

        Args:
            image: [H, W] numpy array

        Returns:
            roi_mask: [H, W] numpy array with 1 for ROI, 0 for background
        """
        # Get image shape
        H, W = image.shape

        # Initialize ROI mask (entire image is ROI)
        roi_mask = np.ones((H, W), dtype=np.float32)

        # Avoid borders (30 pixels from each edge)
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
            dict with keys:
                - x_normal: [1, H, W] original normal image
                - x_lesion: [1, H, W] image with synthetic lesion
                - mask_lesion: [1, H, W] lesion mask
        """
        # Load normal image
        image_path = self.image_paths[idx]
        x_normal = self._load_and_preprocess(image_path)

        # Ensure x_normal is 2D
        if x_normal.ndim != 2:
            raise ValueError(f"Image should be 2D, got shape {x_normal.shape}")

        # Add synthetic lesion with probability
        if random.random() < self.lesion_prob:
            # Create ROI mask
            roi_mask = self._create_roi_mask(x_normal)

            # Synthesize lesion
            try:
                x_lesion, mask_lesion, _, _ = self.lesion_synthesizer.synthesize(
                    x_normal.copy(),
                    roi_mask,
                    lesion_type='random'  # random, radiolucent, mixed
                )
            except Exception as e:
                # If synthesis fails, use original without lesion
                print(f"[Warning] Lesion synthesis failed for {image_path}: {e}")
                x_lesion = x_normal.copy()
                mask_lesion = np.zeros_like(x_normal)
        else:
            # No lesion (for prototype learning)
            x_lesion = x_normal.copy()
            mask_lesion = np.zeros_like(x_normal)

        # Convert to torch tensors and add channel dimension
        x_normal = torch.from_numpy(x_normal).unsqueeze(0).float()  # [1, H, W]
        x_lesion = torch.from_numpy(x_lesion).unsqueeze(0).float()  # [1, H, W]
        mask_lesion = torch.from_numpy(mask_lesion).unsqueeze(0).float()  # [1, H, W]

        return {
            'x_normal': x_normal,      # Ground truth (original)
            'x_lesion': x_lesion,      # Input (with synthetic lesion)
            'mask_lesion': mask_lesion, # Lesion location
        }


class AbnormalDataset(Dataset):
    """
    Real Abnormal Image Dataset for Testing

    실제 병변이 있는 이미지를 테스트
    - 모델이 알아서 병변 위치를 찾고 재구성
    - Ground truth mask는 optional (평가용)

    Directory structure:
        data_root/
            images/
                abnormal_001.png
                abnormal_002.png
            masks/ (optional)
                abnormal_001.png
                abnormal_002.png
    """

    def __init__(self, data_root, image_size=256, has_masks=False,
                 window_center=2048, window_width=4096):
        """
        Args:
            data_root: Root directory containing abnormal images
            image_size: Target image size
            has_masks: Whether ground truth masks are available
            window_center: Window level center for DICOM
            window_width: Window level width for DICOM
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.has_masks = has_masks
        self.window_center = window_center
        self.window_width = window_width

        # Image and mask directories
        self.image_dir = self.data_root / 'images'
        self.mask_dir = self.data_root / 'masks' if has_masks else None

        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")

        if has_masks and not self.mask_dir.exists():
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # Load image paths
        self.image_paths = self._load_image_paths()

        print(f"[AbnormalDataset] Loaded {len(self.image_paths)} abnormal images")
        print(f"[AbnormalDataset] Ground truth masks: {'Available' if has_masks else 'Not available'}")

    def _load_image_paths(self):
        """Load all image paths from images directory"""
        image_paths = []

        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.dcm', '*.dicom']

        for ext in extensions:
            image_paths.extend(list(self.image_dir.glob(ext)))

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        return image_paths

    def _apply_window_level(self, image, window_center, window_width):
        """Apply window level normalization (same as DSRNDataset)"""
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        image = np.clip(image, img_min, img_max)
        image = (image - img_min) / (img_max - img_min)

        return image.astype(np.float32)

    def _load_and_preprocess(self, image_path):
        """Load and preprocess image (supports both standard images and DICOM)"""
        image_path = Path(image_path)

        # Check if DICOM file
        if image_path.suffix.lower() in ['.dcm', '.dicom']:
            if not PYDICOM_AVAILABLE:
                raise ImportError("pydicom is required. Install with: pip install pydicom")

            dcm = pydicom.dcmread(str(image_path))
            image = dcm.pixel_array.astype(np.float32)
            image = self._apply_window_level(image, self.window_center, self.window_width)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            image = image.astype(np.float32) / 255.0

        # Resize
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - x_abnormal: [1, H, W] abnormal image (input)
                - mask_gt: [1, H, W] ground truth mask (if available, else zeros)
                - image_name: str, filename
        """
        # Load abnormal image
        image_path = self.image_paths[idx]
        x_abnormal = self._load_and_preprocess(image_path)

        # Load ground truth mask if available
        if self.has_masks:
            mask_path = self.mask_dir / image_path.name

            if mask_path.exists():
                mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                if mask_gt is None:
                    print(f"[Warning] Failed to load mask: {mask_path}, using zeros")
                    mask_gt = np.zeros_like(x_abnormal)
                else:
                    # Resize and normalize
                    if mask_gt.shape != x_abnormal.shape:
                        mask_gt = cv2.resize(mask_gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

                    mask_gt = (mask_gt > 127).astype(np.float32)  # Binary threshold
            else:
                print(f"[Warning] Mask not found for {image_path.name}, using zeros")
                mask_gt = np.zeros_like(x_abnormal)
        else:
            mask_gt = np.zeros_like(x_abnormal)

        # Convert to torch tensors
        x_abnormal = torch.from_numpy(x_abnormal).unsqueeze(0).float()  # [1, H, W]
        mask_gt = torch.from_numpy(mask_gt).unsqueeze(0).float()  # [1, H, W]

        return {
            'x_abnormal': x_abnormal,
            'mask_gt': mask_gt,
            'image_name': image_path.name
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
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
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
        window_center: int = 2048
        window_width: int = 4096

    # Test dataset
    print("Testing DSRNDataset...")
    dataset = DSRNDataset(
        data_root=TestConfig.data_root,
        image_size=TestConfig.image_size,
        lesion_prob=TestConfig.lesion_prob,
        window_center=TestConfig.window_center,
        window_width=TestConfig.window_width
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test one sample
    try:
        sample = dataset[0]

        print(f"\nSample keys: {sample.keys()}")
        print(f"x_normal shape: {sample['x_normal'].shape}")
        print(f"x_lesion shape: {sample['x_lesion'].shape}")
        print(f"mask_lesion shape: {sample['mask_lesion'].shape}")
        print(f"x_normal range: [{sample['x_normal'].min():.3f}, {sample['x_normal'].max():.3f}]")
        print(f"x_lesion range: [{sample['x_lesion'].min():.3f}, {sample['x_lesion'].max():.3f}]")

        print("\nDataset test passed!")
    except Exception as e:
        print(f"\nDataset test failed: {e}")
        import traceback
        traceback.print_exc()
