"""
Training script for PDD (Pruning During Distillation)
This file contains only the training phase
"""

import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm

from utils.data_loader import get_cifar10_dataloaders
from models.resnet import resnet20, resnet56
from utils.trainer import PDDTrainer
from utils.helpers import set_seed, save_checkpoint
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train PDD Model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model parameters
    parser.add_argument('--teacher_checkpoint', type=str, 
                        default='checkpoints/resnet56_cifar10.pth')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    return parser.parse_args()


def download_teacher_checkpoint(checkpoint_path):
    """Download teacher checkpoint from GitHub if not exists"""
    if not os.path.exists(checkpoint_path):
        print(f"Teacher checkpoint not found at {checkpoint_path}")
        print(f"Downloading teacher checkpoint...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        url = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt"
        
        try:
            import urllib.request
            print(f"Downloading from: {url}")
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"✓ Downloaded checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"✗ Error downloading checkpoint: {e}")
            print(f"\nPlease download manually:")
            print(f"URL: {url}")
            print(f"Save to: {checkpoint_path}")
            exit(1)
    else:
        print(f"✓ Teacher checkpoint found at {checkpoint_path}")


def load_teacher_model(checkpoint_path, device):
    """Load teacher model from checkpoint"""
    print(f"\nLoading teacher model from {checkpoint_path}...")
    
    teacher = resnet56(num_classes=10)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if exists (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        teacher.load_state_dict(new_state_dict)
        print("✓ Teacher model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading teacher model: {e}")
        print("Please check the checkpoint file")
        exit(1)
    
    teacher = teacher.to(device)
    teacher.eval()
    
    # Disable gradients for teacher
    for param in teacher.parameters():
        param.requires_grad = False
    
    return teacher


def evaluate_teacher(teacher, test_loader, device):
    """Evaluate teacher model accuracy"""
    print("\nEvaluating teacher model...")
    teacher.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing teacher'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = teacher(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f"✓ Teacher Accuracy: {accuracy:.2f}%")
    
    return accuracy


def main():
    args = parse_args()
    
    # Load configuration
    config = Config.from_args(args)
    config.display()
    
    # Set seed for reproducibility
    set_seed(config.SEED)
    
    # Create directories
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading CIFAR10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Download and load teacher model
    download_teacher_checkpoint(config.TEACHER_CHECKPOINT)
    teacher = load_teacher_model(config.TEACHER_CHECKPOINT, device)
    
    # Evaluate teacher
    teacher_acc = evaluate_teacher(teacher, test_loader, device)
    
    # Create student model
    print(f"\nCreating student model: {config.STUDENT_MODEL}")
    student = resnet20(num_classes=config.NUM_CLASSES)
    student = student.to(device)
    
    # Count parameters
    student_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"✓ Student parameters: {student_params:,}")
    print(f"✓ Teacher parameters: {teacher_params:,}")
    print(f"✓ Compression ratio: {teacher_params/student_params:.2f}x")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            student.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✓ Resumed from epoch {start_epoch}")
        else:
            print(f"✗ No checkpoint found at {args.resume}")
    
    # Create trainer with config
    print("\n" + "="*60)
    print("Initializing PDD Trainer")
    print("="*60)
    
    trainer = PDDTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        args=args
    )
    
    # Training phase
    print("\n" + "="*60)
    print("Phase 1: Pruning During Distillation")
    print("="*60)
    print(f"Training for {config.DISTILL_EPOCHS} epochs")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Temperature: {config.TEMPERATURE}")
    print(f"Alpha: {config.ALPHA}")
    print("="*60 + "\n")
    
    best_acc = trainer.train()
    
    # Save final model
    checkpoint_path = os.path.join(config.SAVE_DIR, 'student_with_masks.pth')
    save_checkpoint({
        'epoch': config.DISTILL_EPOCHS,
        'state_dict': student.state_dict(),
        'masks': trainer.get_masks(),
        'best_acc': best_acc,
        'teacher_acc': teacher_acc,
        'config': vars(config)
    }, checkpoint_path)
    
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    print(f"Best Student Accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
