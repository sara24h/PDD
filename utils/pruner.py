import torch
import torch.nn as nn
import argparse
import os
from utils.data_loader import get_cifar10_dataloaders
from models.resnet import resnet20, resnet56
from utils.trainer import PDDTrainer
from utils.pruner import ModelPruner
from utils.helpers import set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='PDD: Pruning During Distillation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # Model parameters
    parser.add_argument('--student_model', type=str, default='resnet20', help='Student model')
    parser.add_argument('--teacher_model', type=str, default='resnet56', help='Teacher model')
    parser.add_argument('--teacher_checkpoint', type=str, 
                        default='checkpoints/resnet56_cifar10.pth', 
                        help='Teacher model checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of distillation epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--lr_decay_epochs', type=list, default=[20, 40], help='LR decay epochs')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='LR decay rate')
    
    # Distillation parameters
    parser.add_argument('--temperature', type=float, default=4.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for distillation loss')
    
    # Fine-tuning parameters
    parser.add_argument('--finetune_epochs', type=int, default=100, help='Fine-tuning epochs')
    parser.add_argument('--finetune_lr', type=float, default=0.01, help='Fine-tuning learning rate')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    return parser.parse_args()


def convert_state_dict_keys(state_dict):
    """Convert checkpoint keys to match model architecture"""
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert fc to linear
        if 'fc.' in key:
            new_key = key.replace('fc.', 'linear.')
        
        # Convert downsample to shortcut
        elif 'downsample.' in key:
            new_key = key.replace('downsample.', 'shortcut.')
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def download_teacher_checkpoint(checkpoint_path):
    """Download teacher checkpoint from GitHub if not exists"""
    if not os.path.exists(checkpoint_path):
        print(f"Downloading teacher checkpoint...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # URL for pretrained ResNet56 on CIFAR10
        url = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"Downloaded checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            print("Please download manually from:")
            print(url)
            exit(1)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create student model
    print(f"Creating student model: {args.student_model}")
    student = resnet20(num_classes=10)
    student = student.to(device)
    
    # Load teacher model
    print(f"Loading teacher model: {args.teacher_model}")
    teacher = resnet56(num_classes=10)
    
    # Download teacher checkpoint if needed
    download_teacher_checkpoint(args.teacher_checkpoint)
    
    # Load teacher weights with key conversion
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Convert keys to match model architecture
    state_dict = convert_state_dict_keys(state_dict)
    
    # Load with strict=False to handle any remaining mismatches
    teacher.load_state_dict(state_dict, strict=False)
    
    teacher = teacher.to(device)
    teacher.eval()
    
    print("Teacher model loaded successfully")
    
    # Create trainer
    print("Initializing PDD Trainer...")
    trainer = PDDTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        args=args
    )
    
    # Phase 1: Pruning During Distillation
    print("\n" + "="*50)
    print("Phase 1: Pruning During Distillation")
    print("="*50)
    trainer.train()
    
    # Save student with masks
    checkpoint_path = os.path.join(args.save_dir, 'student_with_masks.pth')
    save_checkpoint({
        'state_dict': student.state_dict(),
        'masks': trainer.get_masks(),
        'args': args
    }, checkpoint_path)
    print(f"\nSaved student with masks to {checkpoint_path}")
    
    # Phase 2: Prune the model
    print("\n" + "="*50)
    print("Phase 2: Structural Pruning")
    print("="*50)
    
    # Create a clean student model without MaskedConv2d wrappers
    clean_student = resnet20(num_classes=10).to(device)
    
    # Extract weights from wrapped model
    wrapped_state = student.state_dict()
    clean_state = {}
    
    for key in wrapped_state.keys():
        # Remove .conv. from MaskedConv2d layers
        clean_key = key.replace('.conv.', '.')
        clean_state[clean_key] = wrapped_state[key]
    
    # Load cleaned state dict
    clean_student.load_state_dict(clean_state, strict=False)
    
    # Get binary masks
    binary_masks = trainer.get_masks()
    
    # Apply structural pruning
    pruner = ModelPruner(clean_student, binary_masks)
    pruned_student = pruner.prune()
    
    # Calculate compression statistics
    original_params, pruned_params = pruner.get_params_count()
    original_flops, pruned_flops = pruner.get_flops_count()
    
    params_reduction = (1 - pruned_params / original_params) * 100
    flops_reduction = (1 - pruned_flops / original_flops) * 100
    
    print(f"\nCompression Statistics:")
    print(f"Original Parameters: {original_params:,}")
    print(f"Pruned Parameters: {pruned_params:,}")
    print(f"Parameters Reduction: {params_reduction:.2f}%")
    print(f"Original FLOPs: {original_flops:,}")
    print(f"Pruned FLOPs: {pruned_flops:,}")
    print(f"FLOPs Reduction: {flops_reduction:.2f}%")
    
    # Phase 3: Fine-tune pruned model
    print("\n" + "="*50)
    print("Phase 3: Fine-tuning Pruned Model")
    print("="*50)
    
    pruned_student = pruned_student.to(device)
    
    # Create optimizer for fine-tuning
    optimizer = torch.optim.SGD(
        pruned_student.parameters(),
        lr=args.finetune_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 80],
        gamma=args.lr_decay_rate
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(args.finetune_epochs):
        # Train
        pruned_student.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = pruned_student(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        
        # Evaluate
        pruned_student.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pruned_student(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        
        print(f"Epoch [{epoch+1}/{args.finetune_epochs}] "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': pruned_student.state_dict(),
                'accuracy': test_acc,
                'params_reduction': params_reduction,
                'flops_reduction': flops_reduction,
                'args': args
            }, os.path.join(args.save_dir, 'pruned_student_best.pth'))
        
        scheduler.step()
    
    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Parameters Reduction: {params_reduction:.2f}%")
    print(f"FLOPs Reduction: {flops_reduction:.2f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
