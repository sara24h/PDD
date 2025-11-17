"""
Fine-tuning script for PDD (Pruning During Distillation)
This file contains only the fine-tuning phase
"""

import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm

from models.resnet import resnet20
from utils.data_loader import get_cifar10_dataloaders
from utils.helpers import set_seed, save_checkpoint, AverageMeter
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Pruned Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pruned model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[60, 80])
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{total_epochs}]')
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Measure accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = 100. * correct / batch_size
        
        # Update metrics
        losses.update(loss.item(), batch_size)
        top1.update(accuracy, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%'
        })
    
    return losses.avg, top1.avg


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Measure accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100. * correct / batch_size
            
            # Update metrics
            losses.update(loss.item(), batch_size)
            top1.update(accuracy, batch_size)
    
    return losses.avg, top1.avg


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Phase 3: Fine-tuning Pruned Model")
    print("="*60)
    
    # Set seed
    set_seed(args.seed)
    
    # Check checkpoint
    if not os.path.isfile(args.checkpoint):
        print(f"✗ Error: Checkpoint not found at {args.checkpoint}")
        exit(1)
    
    print(f"Loading pruned model from: {args.checkpoint}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading CIFAR10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Load pruned model
    print("\nLoading pruned model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model (pruned architecture)
    model = resnet20(num_classes=10)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("✓ Model loaded successfully")
    
    # Display compression statistics if available
    if 'params_reduction' in checkpoint:
        print("\n" + "-"*60)
        print("Compression Statistics:")
        print("-"*60)
        print(f"Parameters Reduction: {checkpoint['params_reduction']:.2f}%")
        print(f"FLOPs Reduction: {checkpoint['flops_reduction']:.2f}%")
        print(f"Original Parameters: {checkpoint.get('original_params', 'N/A'):,}")
        print(f"Pruned Parameters: {checkpoint.get('pruned_params', 'N/A'):,}")
        print("-"*60)
    
    # Count current parameters
    current_params = sum(p.numel() for p in model.parameters())
    print(f"\nCurrent model parameters: {current_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_decay_epochs,
        gamma=args.lr_decay_rate
    )
    
    # Resume if specified
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume and os.path.isfile(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_checkpoint['state_dict'])
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        scheduler.load_state_dict(resume_checkpoint['scheduler'])
        start_epoch = resume_checkpoint['epoch'] + 1
        best_acc = resume_checkpoint['best_acc']
        print(f"✓ Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    # Training loop
    print("\n" + "="*60)
    print(f"Starting fine-tuning for {args.epochs} epochs")
    print(f"Learning rate: {args.lr}")
    print(f"LR decay at epochs: {args.lr_decay_epochs}")
    print("="*60 + "\n")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        if is_best:
            print(f"★ New best accuracy: {best_acc:.2f}%")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0 or is_best:
            checkpoint_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'test_acc': test_acc,
                'train_acc': train_acc
            }
            
            # Add compression stats if available
            if 'params_reduction' in checkpoint:
                checkpoint_dict['params_reduction'] = checkpoint['params_reduction']
                checkpoint_dict['flops_reduction'] = checkpoint['flops_reduction']
                checkpoint_dict['original_params'] = checkpoint.get('original_params')
                checkpoint_dict['pruned_params'] = checkpoint.get('pruned_params')
            
            if is_best:
                save_path = os.path.join(args.save_dir, 'pruned_student_best.pth')
                save_checkpoint(checkpoint_dict, save_path)
            
            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(args.save_dir, f'pruned_student_epoch_{epoch+1}.pth')
                save_checkpoint(checkpoint_dict, save_path)
        
        print("-"*60)
    
    # Final results
    print("\n" + "="*60)
    print("Fine-tuning Completed!")
    print("="*60)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    if 'params_reduction' in checkpoint:
        print(f"Parameters Reduction: {checkpoint['params_reduction']:.2f}%")
        print(f"FLOPs Reduction: {checkpoint['flops_reduction']:.2f}%")
    
    print(f"Best model saved to: {os.path.join(args.save_dir, 'pruned_student_best.pth')}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
