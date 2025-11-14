"""
Dataset with CLAHE + Morphological Gradient preprocessing
for subtle texture/boundary detection in dental X-rays
"""
import os
import numpy as np
import pydicom
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Optional, Tuple, List
import albumentations as A
from sklearn.model_selection import train_test_split
import cv2


class PanoramaXrayDataset(Dataset):
    """
    Panoramic X-ray DICOM dataset with CLAHE + Morphological Gradient.

    Returns:
        - Original image (3ch, normalized to [-1, 1])
        - Morphological gradient map (1ch, in [0, 1])
    """

    def __init__(
            self,
            data_dir: str,
            file_list: Optional[List[Path]] = None,
            window_center: int = 2000,
            window_width: int = 4000,
            target_size: Tuple[int, int] = (256, 256),
            augment: bool = False,
            # CLAHE settings
            use_clahe: bool = True,
            clahe_clip_limit: float = 2.0,
            clahe_tile_size: int = 8,
            # Morphological gradient settings
            use_morph_grad: bool = True,
            morph_kernel_size: int = 3,
            morph_kernel_shape: str = 'ellipse',
            use_multiscale_morph: bool = True,
            morph_scales: Tuple[int, ...] = (3, 5),
            morph_scale_weights: Tuple[float, ...] = (0.6, 0.4),
    ):
        """
        Args:
            data_dir: Directory containing .dcm files
            file_list: Optional list of specific files
            window_center: WL for intensity normalization
            window_width: WW
            target_size: Output size (H, W)
            augment: Whether to apply augmentation
            use_clahe: Apply CLAHE preprocessing
            clahe_clip_limit: CLAHE contrast limit
            clahe_tile_size: CLAHE tile grid size
            use_morph_grad: Compute morphological gradient
            morph_kernel_size: Kernel size for morphology
            morph_kernel_shape: 'ellipse', 'rect', or 'cross'
            use_multiscale_morph: Use multi-scale gradients
            morph_scales: Kernel sizes for multi-scale
            morph_scale_weights: Weights for each scale
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

        # CLAHE settings
        self.use_clahe = use_clahe
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=(clahe_tile_size, clahe_tile_size)
            )
            print(f"[Dataset] CLAHE enabled: clip_limit={clahe_clip_limit}, tile_size={clahe_tile_size}")

        # Morphological gradient settings
        self.use_morph_grad = use_morph_grad
        self.use_multiscale_morph = use_multiscale_morph
        self.morph_scales = morph_scales
        self.morph_scale_weights = morph_scale_weights

        if self.use_morph_grad:
            # Create morphological kernels
            if morph_kernel_shape == 'ellipse':
                shape = cv2.MORPH_ELLIPSE
            elif morph_kernel_shape == 'rect':
                shape = cv2.MORPH_RECT
            elif morph_kernel_shape == 'cross':
                shape = cv2.MORPH_CROSS
            else:
                raise ValueError(f"Unknown kernel shape: {morph_kernel_shape}")

            if use_multiscale_morph:
                self.morph_kernels = [
                    cv2.getStructuringElement(shape, (size, size))
                    for size in morph_scales
                ]
                print(f"[Dataset] Multi-scale morphological gradient: scales={morph_scales}, weights={morph_scale_weights}")
            else:
                self.morph_kernels = [cv2.getStructuringElement(shape, (morph_kernel_size, morph_kernel_size))]
                print(f"[Dataset] Morphological gradient: kernel_size={morph_kernel_size}, shape={morph_kernel_shape}")

        # Augmentation pipeline
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (3, H, W) float32 tensor, normalized to [-1, 1]
            morph_grad: (1, H, W) float32 tensor, in [0, 1]
        """
        dcm_path = self.dicom_files[idx]

        # Read DICOM
        dcm = pydicom.dcmread(str(dcm_path))
        image = dcm.pixel_array.astype(np.float32)

        # Apply window level normalization
        image = self._apply_windowing(image)

        # Normalize to [0, 1] first (for CLAHE)
        image = self._normalize_to_01(image)

        # Apply augmentation & resize (operates on [0, 1])
        transformed = self.transform(image=image)
        image = transformed["image"]  # (H, W) in [0, 1]

        # Compute morphological gradient (before converting to [-1, 1])
        if self.use_morph_grad:
            morph_grad = self._compute_morphological_gradient(image)
        else:
            morph_grad = np.zeros_like(image)

        # Convert image to [-1, 1] for VAE
        image = image * 2.0 - 1.0

        # Convert to tensor
        image = torch.from_numpy(image).float()
        morph_grad = torch.from_numpy(morph_grad).float()

        # Add channel dimensions
        image = image.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        morph_grad = morph_grad.unsqueeze(0)         # (1, H, W)

        return image, morph_grad

    def _apply_windowing(self, image: np.ndarray) -> np.ndarray:
        """Apply window level/width normalization"""
        lower = self.window_center - self.window_width / 2
        upper = self.window_center + self.window_width / 2
        image = np.clip(image, lower, upper)
        return image

    def _normalize_to_01(self, image: np.ndarray) -> np.ndarray:
        """Normalize windowed image to [0, 1]"""
        lower = self.window_center - self.window_width / 2
        upper = self.window_center + self.window_width / 2
        image = (image - lower) / (upper - lower)
        return image.astype(np.float32)

    def _compute_morphological_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Compute morphological gradient after CLAHE enhancement.

        Args:
            image: (H, W) in [0, 1]

        Returns:
            grad: (H, W) in [0, 1]
        """
        # Convert to uint8 for CLAHE and morphology
        image_uint8 = (image * 255).astype(np.uint8)

        # Apply CLAHE if enabled
        if self.use_clahe:
            image_uint8 = self.clahe.apply(image_uint8)

        # Compute morphological gradient: dilation - erosion
        if self.use_multiscale_morph:
            # Multi-scale gradient
            gradients = []
            for kernel in self.morph_kernels:
                dilated = cv2.dilate(image_uint8, kernel)
                eroded = cv2.erode(image_uint8, kernel)
                grad = dilated.astype(np.float32) - eroded.astype(np.float32)
                gradients.append(grad)

            # Weighted combination
            grad = np.zeros_like(gradients[0])
            for g, w in zip(gradients, self.morph_scale_weights):
                grad += w * g
        else:
            # Single-scale gradient
            dilated = cv2.dilate(image_uint8, self.morph_kernels[0])
            eroded = cv2.erode(image_uint8, self.morph_kernels[0])
            grad = dilated.astype(np.float32) - eroded.astype(np.float32)

        # Normalize to [0, 1]
        grad = grad / 255.0

        return grad.astype(np.float32)


def create_train_val_split(
        data_dir: str,
        val_ratio: float = 0.1,
        seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Create train/val split from a single directory.

    Args:
        data_dir: Directory containing all .dcm files
        val_ratio: Fraction of data for validation
        seed: Random seed

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
    import matplotlib.pyplot as plt

    # Test with real data
    print("Dataset module loaded successfully!")
    print("Example usage:")
    print("""
    train_files, val_files = create_train_val_split(
        data_dir="/path/to/dicom/folder",
        val_ratio=0.1,
    )

    dataset = PanoramaXrayDataset(
        data_dir="/path/to/dicom/folder",
        file_list=train_files,
        use_clahe=True,
        use_morph_grad=True,
        use_multiscale_morph=True,
    )

    image, morph_grad = dataset[0]
    # image: (3, 256, 256) in [-1, 1]
    # morph_grad: (1, 256, 256) in [0, 1]
    """)
