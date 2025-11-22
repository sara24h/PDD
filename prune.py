import torch
import argparse
import os
from models.resnet import resnet20
from utils.pruner import ModelPruner
from utils.helpers import save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Prune PDD Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint with masks')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save pruned model')

    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Phase 2: Model Pruning")
    print("="*60)
    
    # Check if checkpoint exists
    if not os.path.isfile(args.checkpoint):
        print(f"✗ Error: Checkpoint not found at {args.checkpoint}")
        print("\nPlease run train.py first to generate the checkpoint with masks")
        exit(1)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'masks' not in checkpoint:
        print("✗ Error: No masks found in checkpoint!")
        print("Please make sure the checkpoint is from training phase")
        exit(1)
    
    print("✓ Checkpoint loaded successfully")
    
    # Create student model
    print("\nCreating student model...")
    student = resnet20(num_classes=10)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        student.load_state_dict(checkpoint['state_dict'])
    else:
        student.load_state_dict(checkpoint)
    
    print("✓ Model weights loaded")
    
    # Get masks
    masks = checkpoint['masks']
    print(f"✓ Loaded {len(masks)} masks")
    
    # Display mask statistics
    print("\n" + "-"*60)
    print("Mask Statistics:")
    print("-"*60)
    
    total_channels = 0
    kept_channels = 0
    
    for name, mask in masks.items():
        mask_flat = mask.squeeze()

        mask_binary = (mask_flat >= -1.0).float()
        total = mask_binary.numel()
        kept = mask_binary.sum().item()
        
        total_channels += total
        kept_channels += kept
        
        pruning_ratio = (1 - kept / total) * 100
        print(f"{name:40s} | Total: {total:4d} | Kept: {kept:4d} | Pruned: {pruning_ratio:5.2f}%")
    
    overall_pruning = (1 - kept_channels / total_channels) * 100
    print("-"*60)
    print(f"{'Overall':40s} | Total: {total_channels:4d} | Kept: {kept_channels:4d} | Pruned: {overall_pruning:5.2f}%")
    print("-"*60)
    
    # Create pruner
    print("\nInitializing pruner...")
    pruner = ModelPruner(student, masks)
    
    # Calculate original statistics
    original_params, pruned_params_est = pruner.get_params_count()
    original_flops, pruned_flops_est = pruner.get_flops_count()
    
    print("\n" + "="*60)
    print("Compression Statistics (Estimated):")
    print("="*60)
    print(f"Original Parameters: {original_params:,}")
    print(f"Pruned Parameters: {pruned_params_est:,}")
    print(f"Parameters Reduction: {(1 - pruned_params_est/original_params)*100:.2f}%")
    print(f"\nOriginal FLOPs: {original_flops:,}")
    print(f"Pruned FLOPs: {pruned_flops_est:,}")
    print(f"FLOPs Reduction: {(1 - pruned_flops_est/original_flops)*100:.2f}%")
    print("="*60)
    
    # Prune the model
    print("\nPruning model...")
    print("-"*60)
    pruned_student = pruner.prune()
    print("-"*60)
    print("✓ Model pruned successfully")
    
    # Calculate actual statistics after pruning
    actual_params = sum(p.numel() for p in pruned_student.parameters())
    params_reduction = (1 - actual_params / original_params) * 100
    flops_reduction = (1 - pruned_flops_est / original_flops) * 100
    
    print("\n" + "="*60)
    print("Final Compression Statistics:")
    print("="*60)
    print(f"Original Parameters: {original_params:,}")
    print(f"Pruned Parameters: {actual_params:,}")
    print(f"Parameters Reduction: {params_reduction:.2f}%")
    print(f"\nOriginal FLOPs: {original_flops:,}")
    print(f"Pruned FLOPs (estimated): {pruned_flops_est:,}")
    print(f"FLOPs Reduction: {flops_reduction:.2f}%")
    print("="*60)
    
    # Save pruned model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'pruned_student.pth')
    
    save_checkpoint({
        'state_dict': pruned_student.state_dict(),
        'params_reduction': params_reduction,
        'flops_reduction': flops_reduction,
        'original_params': original_params,
        'pruned_params': actual_params,
        'original_flops': original_flops,
        'pruned_flops': pruned_flops_est,
        'source_checkpoint': args.checkpoint
    }, save_path)
    
    print(f"\n✓ Pruned model saved to: {save_path}")
    print("\nNext step: Run finetune.py to fine-tune the pruned model")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
