import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from models.masked_resnet import MaskedResNet20


def visualize_channel_importance(model, save_path='pruning_visualization.png'):
    """
    Visualize channel importance scores across all layers.
    Similar to Figure 5 in the paper.
    """
    # Get channel scores for all layers
    scores_dict = model.get_all_channel_scores()
    
    if not scores_dict:
        print("No masked layers found in the model!")
        return
    
    # Prepare data for plotting
    layer_names = []
    all_scores = []
    
    for name, scores in scores_dict.items():
        if len(scores) > 0:
            layer_names.append(name.replace('module.', ''))
            all_scores.append(scores.cpu().numpy())
    
    # Create figure
    fig, axes = plt.subplots(len(all_scores), 1, figsize=(12, 2*len(all_scores)))
    
    if len(all_scores) == 1:
        axes = [axes]
    
    for idx, (name, scores) in enumerate(zip(layer_names, all_scores)):
        ax = axes[idx]
        
        # Plot channel importance scores
        channels = np.arange(len(scores))
        colors = ['red' if s < 0.5 else 'blue' for s in scores]
        
        ax.bar(channels, scores, color=colors, alpha=0.7)
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Threshold')
        
        ax.set_ylabel('Importance', fontsize=10)
        ax.set_title(f'Layer: {name}', fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        num_pruned = sum(1 for s in scores if s < 0.5)
        prune_ratio = num_pruned / len(scores) * 100
        ax.text(0.98, 0.95, f'Pruned: {num_pruned}/{len(scores)} ({prune_ratio:.1f}%)',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Channel Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def visualize_pruning_evolution(checkpoint_paths, save_path='pruning_evolution.png'):
    """
    Visualize how pruning evolves across different epochs.
    Similar to Figure 4 in the paper.
    """
    epochs = []
    pruning_ratios = []
    accuracies = []
    
    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path)
            epoch = checkpoint.get('epoch', 0)
            acc = checkpoint.get('accuracy', 0)
            
            # Load model
            model = MaskedResNet20()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get pruning info
            info = model.get_pruning_info()
            ratio = info['overall']['ratio'] * 100
            
            epochs.append(epoch)
            pruning_ratios.append(ratio)
            accuracies.append(acc)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    if not epochs:
        print("No valid checkpoints found!")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot accuracy progression
    ax1.plot(epochs, accuracies, 'b-o', linewidth=2, markersize=8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Progression Over Training', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(epochs)+1])
    
    # Plot pruning ratio progression
    ax2.plot(epochs, pruning_ratios, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pruning Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Pruning Ratio Progression', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(epochs)+1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evolution visualization saved to {save_path}")
    plt.close()


def visualize_layer_wise_pruning(model, save_path='layer_wise_pruning.png'):
    """
    Visualize pruning statistics for each layer.
    Similar to Figure 1 in the paper.
    """
    pruning_info = model.get_pruning_info()
    
    if not pruning_info or 'overall' not in pruning_info:
        print("No pruning information available!")
        return
    
    # Extract layer-wise data
    layers = []
    total_channels = []
    kept_channels = []
    pruned_channels = []
    
    for name, info in pruning_info.items():
        if name != 'overall':
            layers.append(name.replace('module.', '').split('.')[-1])
            total_channels.append(info['total'])
            kept_channels.append(info['kept'])
            pruned_channels.append(info['pruned'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, total_channels, width, 
                   label='Original Channels', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, kept_channels, width, 
                   label='Kept Channels', color='darkblue', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Channels', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Channel Pruning (Before vs After)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add overall statistics
    overall = pruning_info['overall']
    stats_text = (f"Overall Statistics:\n"
                 f"Total Channels: {overall['total']}\n"
                 f"Pruned Channels: {overall['pruned']}\n"
                 f"Pruning Ratio: {overall['ratio']*100:.2f}%")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Layer-wise visualization saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Pruning Results')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_masked_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--evolution-checkpoints', type=str, nargs='+',
                       help='Paths to checkpoints at different epochs')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    model = MaskedResNet20()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nGenerating visualizations...")
    
    # Visualize channel importance
    print("1. Channel importance scores...")
    visualize_channel_importance(
        model, 
        save_path=os.path.join(args.output_dir, 'channel_importance.png')
    )
    
    # Visualize layer-wise pruning
    print("2. Layer-wise pruning...")
    visualize_layer_wise_pruning(
        model,
        save_path=os.path.join(args.output_dir, 'layer_wise_pruning.png')
    )
    
    # Visualize evolution if checkpoints provided
    if args.evolution_checkpoints:
        print("3. Pruning evolution...")
        visualize_pruning_evolution(
            args.evolution_checkpoints,
            save_path=os.path.join(args.output_dir, 'pruning_evolution.png')
        )
    
    print(f"\nAll visualizations saved to {args.output_dir}/")
    
    # Print pruning statistics
    print("\n" + "="*60)
    print("PRUNING STATISTICS")
    print("="*60)
    
    pruning_info = model.get_pruning_info()
    overall = pruning_info['overall']
    
    print(f"\nOverall:")
    print(f"  Total Channels: {overall['total']}")
    print(f"  Kept Channels: {overall['total'] - overall['pruned']}")
    print(f"  Pruned Channels: {overall['pruned']}")
    print(f"  Pruning Ratio: {overall['ratio']*100:.2f}%")
    
    print(f"\nLayer-wise Details:")
    for name, info in pruning_info.items():
        if name != 'overall':
            print(f"  {name}:")
            print(f"    Total: {info['total']}, Kept: {info['kept']}, "
                  f"Pruned: {info['pruned']} ({info['ratio']*100:.1f}%)")


if __name__ == '__main__':
    main()
