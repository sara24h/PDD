import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from models.resnet import resnet20


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize PDD Masks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint with masks')
    parser.add_argument('--save_path', type=str, default='mask_visualization.png',
                        help='Path to save visualization')
    return parser.parse_args()


def visualize_masks(checkpoint_path, save_path):
    """Visualize learned masks"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'masks' not in checkpoint:
        print("No masks found in checkpoint!")
        return
    
    masks = checkpoint['masks']
    
    # Prepare data for visualization
    layer_names = []
    original_channels = []
    kept_channels = []
    pruning_ratios = []
    
    for name, mask in masks.items():
        mask_binary = (mask.squeeze() > 0.5).float()
        total = mask_binary.numel()
        kept = mask_binary.sum().item()
        pruned = total - kept
        pruning_ratio = (pruned / total) * 100
        
        layer_names.append(name.split('.')[-1])  # Get layer name
        original_channels.append(total)
        kept_channels.append(kept)
        pruning_ratios.append(pruning_ratio)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Channel counts
    x = np.arange(len(layer_names))
    width = 0.35
    
    ax1.bar(x - width/2, original_channels, width, label='Original', alpha=0.7)
    ax1.bar(x + width/2, kept_channels, width, label='After Pruning', alpha=0.7)
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Channels')
    ax1.set_title('Channel Count Before and After Pruning')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Pruning ratios
    colors = ['red' if r > 50 else 'orange' if r > 30 else 'green' 
              for r in pruning_ratios]
    
    ax2.bar(x, pruning_ratios, color=colors, alpha=0.7)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Pruning Ratio (%)')
    ax2.set_title('Pruning Ratio per Layer')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names, rotation=45, ha='right')
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50%')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Pruning Statistics per Layer:")
    print("="*60)
    print(f"{'Layer':<20} {'Original':<12} {'Kept':<12} {'Pruned%':<12}")
    print("-"*60)
    
    for i, name in enumerate(layer_names):
        print(f"{name:<20} {original_channels[i]:<12} {kept_channels[i]:<12} "
              f"{pruning_ratios[i]:<12.2f}")
    
    print("-"*60)
    avg_pruning = np.mean(pruning_ratios)
    print(f"{'Average':<20} {'':<12} {'':<12} {avg_pruning:<12.2f}")
    print("="*60)


def main():
    args = parse_args()
    visualize_masks(args.checkpoint, args.save_path)


if __name__ == '__main__':
    main()
