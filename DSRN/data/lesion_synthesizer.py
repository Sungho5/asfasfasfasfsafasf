"""
Diverse Lesion Synthesizer: Radiolucent Only with Various Shapes
다양한 형태와 파괴 강도를 가진 투과성 병변만 생성
"""

import cv2
import numpy as np
import random
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter


class DiverseLesionSynthesizer:
    """다양한 형태의 투과성 병변 합성기"""

    def __init__(self):
        self.shape_types = [
            'circular',
            'elliptical',
            'irregular',
            'multilocular',
            'root_resorption'
        ]

        # Lesion destruction types
        self.destruction_types = [
            'mild',           # 약한 밝기 감소 + blur
            'moderate',       # 중간 밝기 감소 + texture 파괴
            'severe',         # 심한 밝기 감소 + 거의 0에 가깝게
            'complete_hole',  # 아예 구멍 뚫기 (0으로 만들기)
            'texture_破壞'    # Texture pattern 완전 파괴
        ]

    def synthesize(self, image, num_lesions=None):
        """
        병변을 합성하여 이미지에 추가

        Args:
            image: numpy array (H, W) or (H, W, 1), 0-1 normalized
            num_lesions: 생성할 병변 개수 (None이면 랜덤)

        Returns:
            lesion_image: 병변이 추가된 이미지
            mask: 병변 위치 마스크 (0-1)
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Handle shape
        if image.ndim == 3:
            image = image.squeeze(-1)

        h, w = image.shape

        # Random number of lesions
        if num_lesions is None:
            num_lesions = random.randint(1, 3)

        # Initialize
        lesion_image = image.copy()
        total_mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(num_lesions):
            # Random shape type
            shape_type = random.choice(self.shape_types)

            # Random location (avoid edges)
            center_x = random.randint(w // 4, 3 * w // 4)
            center_y = random.randint(h // 4, 3 * h // 4)

            # Random size
            radius = random.randint(15, 50)

            # Generate lesion
            if shape_type == 'circular':
                lesion_mask = self._create_circular(h, w, center_x, center_y, radius)
            elif shape_type == 'elliptical':
                lesion_mask = self._create_elliptical(h, w, center_x, center_y, radius)
            elif shape_type == 'irregular':
                lesion_mask = self._create_irregular(h, w, center_x, center_y, radius)
            elif shape_type == 'multilocular':
                lesion_mask = self._create_multilocular(h, w, center_x, center_y, radius)
            else:  # root_resorption
                lesion_mask = self._create_root_resorption(h, w, center_x, center_y, radius)

            # Random destruction type
            destruction_type = random.choice(self.destruction_types)

            # Apply lesion effect
            lesion_image = self._apply_lesion_effect(lesion_image, lesion_mask, destruction_type)

            # Accumulate mask
            total_mask = np.maximum(total_mask, lesion_mask)

        return lesion_image, total_mask

    def _create_circular(self, h, w, cx, cy, radius):
        """원형 병변"""
        y, x = np.ogrid[:h, :w]
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2).astype(np.float32)

        # Smooth boundary
        mask = gaussian_filter(mask, sigma=2.0)

        return mask

    def _create_elliptical(self, h, w, cx, cy, radius):
        """타원형 병변"""
        y, x = np.ogrid[:h, :w]

        # Random aspect ratio and rotation
        aspect = random.uniform(0.5, 1.5)
        angle = random.uniform(0, 180)

        # Rotate coordinates
        angle_rad = np.radians(angle)
        x_rot = (x - cx) * np.cos(angle_rad) + (y - cy) * np.sin(angle_rad)
        y_rot = -(x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad)

        mask = ((x_rot / radius) ** 2 + (y_rot / (radius * aspect)) ** 2 <= 1).astype(np.float32)

        # Smooth boundary
        mask = gaussian_filter(mask, sigma=2.0)

        return mask

    def _create_irregular(self, h, w, cx, cy, radius):
        """불규칙한 형태 병변"""
        # Start with circular
        mask = self._create_circular(h, w, cx, cy, radius)

        # Add noise to boundary
        noise = np.random.randn(h, w) * 0.3
        noise = gaussian_filter(noise, sigma=5.0)

        mask = mask + noise
        mask = np.clip(mask, 0, 1)

        # Threshold and smooth
        mask = (mask > 0.5).astype(np.float32)
        mask = gaussian_filter(mask, sigma=2.0)

        return mask

    def _create_multilocular(self, h, w, cx, cy, radius):
        """다방성 병변 (여러 개의 작은 낭종이 모인 형태)"""
        num_lobes = random.randint(2, 4)

        total_mask = np.zeros((h, w), dtype=np.float32)

        for i in range(num_lobes):
            # Random offset from center
            offset_x = random.randint(-radius // 2, radius // 2)
            offset_y = random.randint(-radius // 2, radius // 2)

            lobe_cx = cx + offset_x
            lobe_cy = cy + offset_y
            lobe_radius = radius // 2 + random.randint(-5, 5)

            # Create lobe
            lobe_mask = self._create_circular(h, w, lobe_cx, lobe_cy, lobe_radius)

            total_mask = np.maximum(total_mask, lobe_mask)

        return total_mask

    def _create_root_resorption(self, h, w, cx, cy, radius):
        """치근 흡수 형태 (Root resorption-like)"""
        # Elongated shape
        y, x = np.ogrid[:h, :w]

        # Vertical elongation
        mask = ((x - cx) ** 2 / (radius * 0.5) ** 2 +
                (y - cy) ** 2 / radius ** 2 <= 1).astype(np.float32)

        # Add irregularity at top
        noise = np.random.randn(h, w) * 0.2
        noise = gaussian_filter(noise, sigma=3.0)

        mask = mask + noise * (y < cy).astype(np.float32)
        mask = np.clip(mask, 0, 1)

        # Smooth
        mask = gaussian_filter(mask, sigma=2.0)

        return mask

    def _apply_lesion_effect(self, image, mask, destruction_type='moderate'):
        """
        병변 효과 적용 - 다양한 파괴 타입

        Args:
            image: 원본 이미지
            mask: 병변 마스크 (0-1)
            destruction_type: 'mild', 'moderate', 'severe', 'complete_hole', 'texture_破壞'

        Returns:
            lesion_image: 병변이 적용된 이미지
        """
        h, w = image.shape
        lesion_image = image.copy()

        if destruction_type == 'mild':
            # 약한 밝기 감소 + blur
            intensity_reduction = random.uniform(0.2, 0.4)
            blurred = gaussian_filter(image, sigma=2.0)

            lesion_image = image * (1 - mask * intensity_reduction)
            lesion_image = lesion_image * (1 - mask) + blurred * mask * 0.8

        elif destruction_type == 'moderate':
            # 중간 밝기 감소 + texture 파괴
            intensity_reduction = random.uniform(0.4, 0.6)
            blurred = gaussian_filter(image, sigma=3.0)

            lesion_image = image * (1 - mask * intensity_reduction)
            lesion_image = lesion_image * (1 - mask) + blurred * mask * 0.6

            # Add noise
            noise = np.random.randn(h, w) * 0.03
            lesion_image = lesion_image + noise * mask

        elif destruction_type == 'severe':
            # 심한 파괴 - 거의 0에 가깝게
            intensity_reduction = random.uniform(0.7, 0.9)
            blurred = gaussian_filter(image, sigma=4.0)

            # 거의 검게 만들기
            lesion_image = image * (1 - mask * intensity_reduction)

            # 약간의 blurred texture만 남기기
            lesion_image = lesion_image * (1 - mask) + blurred * mask * 0.2

        elif destruction_type == 'complete_hole':
            # 완전한 구멍 - 아예 0으로 만들기
            # 중심부는 완전히 0, 가장자리는 점진적으로
            center_mask = (mask > 0.7).astype(np.float32)
            edge_mask = mask - center_mask

            # 중심부는 0
            lesion_image = image * (1 - center_mask)

            # 가장자리는 점진적으로 감소
            lesion_image = lesion_image * (1 - edge_mask * 0.8)

        elif destruction_type == 'texture_破壞':
            # Texture pattern 완전 파괴
            # 1. 원본 texture 제거
            very_blurred = gaussian_filter(image, sigma=5.0)

            # 2. Random noise pattern 추가 (broken trabecular 시뮬레이션)
            noise_pattern = np.random.randn(h, w) * 0.1
            noise_pattern = gaussian_filter(noise_pattern, sigma=1.5)

            # 3. 밝기 감소
            intensity_reduction = random.uniform(0.5, 0.7)

            # Combine
            destroyed_texture = very_blurred * (1 - intensity_reduction) + noise_pattern * 0.3
            destroyed_texture = np.clip(destroyed_texture, 0, 1)

            # Apply to lesion area
            lesion_image = image * (1 - mask) + destroyed_texture * mask

        else:
            # Default to moderate
            return self._apply_lesion_effect(image, mask, 'moderate')

        # Final clip
        lesion_image = np.clip(lesion_image, 0, 1)

        return lesion_image

    def synthesize_batch(self, images, num_lesions=None):
        """
        Batch processing

        Args:
            images: torch.Tensor (B, C, H, W) or (B, H, W)
            num_lesions: int or None

        Returns:
            lesion_images: torch.Tensor (B, C, H, W)
            masks: torch.Tensor (B, 1, H, W)
        """
        device = images.device
        dtype = images.dtype

        # Handle shape
        if images.ndim == 3:
            images = images.unsqueeze(1)  # Add channel dim

        B, C, H, W = images.shape

        lesion_images = []
        masks = []

        for i in range(B):
            img = images[i, 0].cpu().numpy()  # (H, W)

            # Synthesize
            lesion_img, mask = self.synthesize(img, num_lesions)

            lesion_images.append(lesion_img)
            masks.append(mask)

        # Convert back to tensor
        lesion_images = torch.tensor(np.array(lesion_images), dtype=dtype, device=device)
        masks = torch.tensor(np.array(masks), dtype=dtype, device=device)

        # Add channel dimension
        lesion_images = lesion_images.unsqueeze(1)  # (B, 1, H, W)
        masks = masks.unsqueeze(1)  # (B, 1, H, W)

        return lesion_images, masks


if __name__ == "__main__":
    """Test - Show all destruction types"""
    import matplotlib.pyplot as plt

    # Create synthesizer
    synthesizer = DiverseLesionSynthesizer()

    # Create dummy image with texture
    test_image = np.random.rand(256, 256) * 0.3 + 0.5
    test_image = gaussian_filter(test_image, sigma=1.0)  # Add some texture

    # Test each destruction type
    destruction_types = ['mild', 'moderate', 'severe', 'complete_hole', 'texture_破壞']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Original
    axes[0].imshow(test_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Each destruction type
    for i, dest_type in enumerate(destruction_types):
        # Create circular mask
        mask = synthesizer._create_circular(256, 256, 128, 128, 40)

        # Apply destruction
        lesion_img = synthesizer._apply_lesion_effect(test_image, mask, dest_type)

        axes[i + 1].imshow(lesion_img, cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f'{dest_type}', fontsize=14, fontweight='bold')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('lesion_destruction_types.png', dpi=150, bbox_inches='tight')
    print("Saved: lesion_destruction_types.png")

    # Test random synthesis
    print("\n[Testing random synthesis with multiple lesions]")
    lesion_image, mask = synthesizer.synthesize(test_image, num_lesions=3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(lesion_image, cmap='gray')
    axes[1].set_title('With Lesions (Random Types)')
    axes[1].axis('off')

    axes[2].imshow(mask, cmap='hot')
    axes[2].set_title('Lesion Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('lesion_synthesis_random.png', dpi=150, bbox_inches='tight')
    print("Saved: lesion_synthesis_random.png")

    print("\n[Test Complete!]")
    print(f"Lesion image range: [{lesion_image.min():.3f}, {lesion_image.max():.3f}]")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
