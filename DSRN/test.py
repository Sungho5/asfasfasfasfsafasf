"""
DSRN Testing Script
학습된 모델로 테스트 및 시각화
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from config import DSRNConfig
from models.dsrn import DSRN
from data.dataset import DSRNDataset
from utils.visualization import visualize_results, save_comparison


class DSRNTester:
    """DSRN Tester"""

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device(config.device)

        # Create model
        self.model = DSRN(config).to(self.device)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)

        print("[Tester] Initialized")

    def load_checkpoint(self, checkpoint_path):
        """Load trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"[Checkpoint] Loaded from {checkpoint_path}")
        print(f"[Checkpoint] Trained epoch: {checkpoint['epoch']}")
        print(f"[Checkpoint] Best val loss: {checkpoint['best_val_loss']:.4f}")

    @torch.no_grad()
    def test_single_image(self, image_path):
        """
        Test on a single image

        Args:
            image_path: Path to input image

        Returns:
            results: Dict containing all outputs
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Preprocess
        original_size = image.shape
        image = cv2.resize(image, (self.config.image_size, self.config.image_size))
        image = image.astype(np.float32) / 255.0

        # To tensor
        x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]

        # Forward
        x_recon, anomaly_map, fusion_weights = self.model(x)

        # Compute residual
        residual = torch.abs(x - x_recon)

        # Convert to numpy
        results = {
            'input': x.cpu().numpy()[0, 0],
            'reconstruction': x_recon.cpu().numpy()[0, 0],
            'anomaly_map': anomaly_map.cpu().numpy()[0, 0],
            'fusion_weights': fusion_weights.cpu().numpy()[0, 0],
            'residual': residual.cpu().numpy()[0, 0],
            'original_size': original_size
        }

        return results

    def test_dataset(self, test_dataset, num_samples=None):
        """
        Test on entire dataset

        Args:
            test_dataset: Dataset to test on
            num_samples: Number of samples to test (None = all)
        """
        self.model.eval()

        output_dir = Path(self.config.output_dir) / 'test_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        num_samples = num_samples or len(test_dataset)
        num_samples = min(num_samples, len(test_dataset))

        print(f"[Test] Testing on {num_samples} samples")

        for idx in tqdm(range(num_samples), desc='Testing'):
            batch = test_dataset[idx]

            x_normal = batch['x_normal'].unsqueeze(0).to(self.device)
            x_lesion = batch['x_lesion'].unsqueeze(0).to(self.device)
            mask_lesion = batch['mask_lesion'].unsqueeze(0).to(self.device)

            # Forward
            x_recon, anomaly_map, fusion_weights = self.model(x_lesion)

            # Save visualization
            save_path = output_dir / f'sample_{idx:04d}.png'
            visualize_results(
                x_normal=x_normal,
                x_lesion=x_lesion,
                x_recon=x_recon,
                mask_lesion=mask_lesion,
                anomaly_map=anomaly_map,
                fusion_weights=fusion_weights,
                save_path=str(save_path)
            )

        print(f"[Test] Results saved to {output_dir}")

    def test_directory(self, image_dir, output_dir=None):
        """
        Test on all images in a directory

        Args:
            image_dir: Directory containing test images
            output_dir: Output directory (default: config.output_dir/test_results)
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir) if output_dir else Path(self.config.output_dir) / 'test_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_paths.extend(list(image_dir.glob(ext)))

        print(f"[Test] Found {len(image_paths)} images in {image_dir}")

        for image_path in tqdm(image_paths, desc='Testing'):
            # Test
            results = self.test_single_image(image_path)

            # Save results
            save_name = image_path.stem
            save_comparison(
                results,
                save_path=output_dir / f'{save_name}_result.png'
            )

        print(f"[Test] Results saved to {output_dir}")

    def compute_metrics(self, test_dataset):
        """
        Compute evaluation metrics

        Args:
            test_dataset: Dataset to evaluate on

        Returns:
            metrics: Dict of metrics
        """
        self.model.eval()

        total_recon_error = 0
        total_anomaly_iou = 0
        num_samples = 0

        for idx in tqdm(range(len(test_dataset)), desc='Computing metrics'):
            batch = test_dataset[idx]

            x_normal = batch['x_normal'].unsqueeze(0).to(self.device)
            x_lesion = batch['x_lesion'].unsqueeze(0).to(self.device)
            mask_lesion = batch['mask_lesion'].unsqueeze(0).to(self.device)

            # Forward
            x_recon, anomaly_map, _ = self.model(x_lesion)

            # Reconstruction error
            recon_error = torch.mean((x_recon - x_normal) ** 2).item()
            total_recon_error += recon_error

            # Anomaly detection IoU (if lesion exists)
            if mask_lesion.sum() > 0:
                # Threshold anomaly map
                anomaly_pred = (anomaly_map > 0.5).float()

                # Compute IoU
                intersection = (anomaly_pred * mask_lesion).sum()
                union = anomaly_pred.sum() + mask_lesion.sum() - intersection
                iou = (intersection / (union + 1e-8)).item()

                total_anomaly_iou += iou
                num_samples += 1

        # Average metrics
        metrics = {
            'reconstruction_mse': total_recon_error / len(test_dataset),
            'anomaly_iou': total_anomaly_iou / max(num_samples, 1)
        }

        print("\n" + "=" * 50)
        print("Evaluation Metrics:")
        print("=" * 50)
        for key, val in metrics.items():
            print(f"{key}: {val:.4f}")
        print("=" * 50)

        return metrics


def main():
    """Main"""
    parser = argparse.ArgumentParser(description='DSRN Testing')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pth',
                        help='Path to checkpoint')
    parser.add_argument('--mode', type=str, default='dataset',
                        choices=['single', 'dataset', 'directory'],
                        help='Test mode')
    parser.add_argument('--input', type=str, default=None,
                        help='Input image or directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test (dataset mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Load config
    config = DSRNConfig()

    # Create tester
    tester = DSRNTester(config, args.checkpoint)

    # Test
    if args.mode == 'single':
        if not args.input:
            raise ValueError("--input is required for single mode")

        results = tester.test_single_image(args.input)

        # Save
        output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir) / 'test_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        save_comparison(results, save_path=output_dir / 'result.png')
        print(f"[Test] Result saved to {output_dir / 'result.png'}")

    elif args.mode == 'dataset':
        # Create test dataset
        test_dataset = DSRNDataset(
            data_root=config.data_root,
            image_size=config.image_size,
            lesion_prob=0.8
        )

        # Test
        tester.test_dataset(test_dataset, num_samples=args.num_samples)

        # Compute metrics
        tester.compute_metrics(test_dataset)

    elif args.mode == 'directory':
        if not args.input:
            raise ValueError("--input is required for directory mode")

        tester.test_directory(args.input, args.output_dir)


if __name__ == "__main__":
    main()
