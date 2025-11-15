"""
Visualization Utilities for DSRN
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def visualize_results(x_normal, x_lesion, x_recon, mask_lesion, anomaly_map, fusion_weights, save_path):
    """
    Visualize DSRN results

    Args:
        x_normal: [B, C, H, W] ground truth - C=3: [original, clahe, gradient]
        x_lesion: [B, C, H, W] input with lesion - C=3
        x_recon: [B, C, H, W] reconstruction - C=3
        mask_lesion: [B, 1, H, W] lesion mask
        anomaly_map: [B, 1, H, W] predicted anomaly
        fusion_weights: [B, 1, H, W] fusion weights
        save_path: Path to save visualization
    """
    # Convert to numpy
    x_normal = tensor_to_numpy(x_normal)
    x_lesion = tensor_to_numpy(x_lesion)
    x_recon = tensor_to_numpy(x_recon)
    mask_lesion = tensor_to_numpy(mask_lesion)
    anomaly_map = tensor_to_numpy(anomaly_map)
    fusion_weights = tensor_to_numpy(fusion_weights)

    # Compute residual (only on first channel - original image)
    residual = np.abs(x_lesion[:, 0:1] - x_recon[:, 0:1])

    # Number of samples
    B = x_normal.shape[0]

    # Create figure
    fig, axes = plt.subplots(B, 7, figsize=(21, 3 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for i in range(B):
        # Original normal
        axes[i, 0].imshow(x_normal[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Original\n(Normal)')
        axes[i, 0].axis('off')

        # Input with lesion
        axes[i, 1].imshow(x_lesion[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Input\n(With Lesion)')
        axes[i, 1].axis('off')

        # Ground truth lesion mask
        axes[i, 2].imshow(mask_lesion[i, 0], cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('GT Lesion\nMask')
        axes[i, 2].axis('off')

        # Predicted anomaly map
        axes[i, 3].imshow(anomaly_map[i, 0], cmap='hot', vmin=0, vmax=1)
        axes[i, 3].set_title('Predicted\nAnomaly Map')
        axes[i, 3].axis('off')

        # Fusion weights
        axes[i, 4].imshow(fusion_weights[i, 0], cmap='viridis', vmin=0, vmax=1)
        axes[i, 4].set_title('Fusion\nWeights')
        axes[i, 4].axis('off')

        # Reconstruction
        axes[i, 5].imshow(x_recon[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 5].set_title('Reconstruction')
        axes[i, 5].axis('off')

        # Residual
        axes[i, 6].imshow(residual[i, 0], cmap='hot', vmin=0, vmax=0.5)
        axes[i, 6].set_title('Residual\n|Input - Recon|')
        axes[i, 6].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Visualization] Saved to {save_path}")


def save_comparison(results, save_path):
    """
    Save comparison for single image

    Args:
        results: Dict with keys: input, reconstruction, anomaly_map, fusion_weights, residual
        save_path: Path to save
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Input
    axes[0].imshow(results['input'], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Input', fontsize=14)
    axes[0].axis('off')

    # Anomaly map
    axes[1].imshow(results['anomaly_map'], cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Anomaly Map', fontsize=14)
    axes[1].axis('off')

    # Fusion weights
    axes[2].imshow(results['fusion_weights'], cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Fusion Weights', fontsize=14)
    axes[2].axis('off')

    # Reconstruction
    axes[3].imshow(results['reconstruction'], cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Reconstruction', fontsize=14)
    axes[3].axis('off')

    # Residual
    axes[4].imshow(results['residual'], cmap='hot', vmin=0, vmax=0.5)
    axes[4].set_title('Residual', fontsize=14)
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Visualization] Saved to {save_path}")


def plot_training_curves(log_dir, save_path=None):
    """
    Plot training curves from tensorboard logs

    Args:
        log_dir: Tensorboard log directory
        save_path: Path to save plot
    """
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get scalar tags
    tags = ea.Tags()['scalars']

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    for tag in ['train/total', 'val/total']:
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            label = 'Train' if 'train' in tag else 'Val'
            axes[0, 0].plot(steps, values, label=label)

    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Reconstruction loss
    for tag in ['train/recon', 'val/recon']:
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            label = 'Train' if 'train' in tag else 'Val'
            axes[0, 1].plot(steps, values, label=label)

    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Anomaly loss
    for tag in ['train/anomaly', 'val/anomaly']:
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            label = 'Train' if 'train' in tag else 'Val'
            axes[1, 0].plot(steps, values, label=label)

    axes[1, 0].set_title('Anomaly Detection Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate
    if 'lr' in tags:
        events = ea.Scalars('lr')
        steps = [e.step for e in events]
        values = [e.value for e in events]

        axes[1, 1].plot(steps, values)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_feature_maps(features, save_path):
    """
    Visualize feature maps

    Args:
        features: Dict with f1, f2, f3, f4
        save_path: Path to save
    """
    features = {k: tensor_to_numpy(v) for k, v in features.items()}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, (name, feat) in enumerate(features.items()):
        if name == 'fused':
            continue

        # Take first sample, average across channels
        feat_avg = feat[0].mean(axis=0)  # [H, W]

        axes[idx].imshow(feat_avg, cmap='viridis')
        axes[idx].set_title(f'{name} ({feat.shape})', fontsize=12)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Visualization] Feature maps saved to {save_path}")


if __name__ == "__main__":
    """Test"""
    import torch

    # Dummy data (multi-channel: C=3)
    B = 2
    x_normal = torch.rand(B, 3, 256, 256)  # 3-channel: [original, clahe, gradient]
    x_lesion = torch.rand(B, 3, 256, 256)
    x_recon = torch.rand(B, 3, 256, 256)
    mask_lesion = torch.rand(B, 1, 256, 256)
    anomaly_map = torch.rand(B, 1, 256, 256)
    fusion_weights = torch.rand(B, 1, 256, 256)

    # Test visualization
    visualize_results(
        x_normal, x_lesion, x_recon,
        mask_lesion, anomaly_map, fusion_weights,
        save_path='test_visualization.png'
    )

    print("Visualization test passed!")
