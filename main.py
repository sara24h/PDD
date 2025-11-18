import torch
import torch.nn as nn
import argparse
import os
import urllib.request
from utils.data_loader import get_cifar10_dataloaders
from models.resnet import resnet20, resnet56
from utils.trainer import PDDTrainer
from utils.pruner import ModelPruner
from utils.helpers import set_seed, save_checkpoint


def download_teacher_checkpoint(checkpoint_path):
    """Download pretrained ResNet56 if not exists"""
    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint found at {checkpoint_path}")
        return True
    
    print(f"✗ Checkpoint not found at {checkpoint_path}")
    print("Downloading pretrained ResNet56 from GitHub...")
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    urls = [
        'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
        'https://github.com/huyvnphan/PyTorch_CIFAR10/releases/download/v3.0.0/resnet56-10b9e6fd.pt'
    ]
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, checkpoint_path)
            
            # Verify
            torch.load(checkpoint_path, map_location='cpu')
            print(f"✓ Successfully downloaded to {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
    
    return False


def load_teacher_model(teacher, checkpoint_path, device):
    """Load teacher model with automatic key mapping"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # تبدیل کلیدها به فرمت سازگار
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # حذف پیشوند module. اگر وجود دارد
        new_key = new_key.replace('module.', '')
        # تبدیل downsample به shortcut
        new_key = new_key.replace('downsample.', 'shortcut.')
        # تبدیل fc به linear
        new_key = new_key.replace('fc.', 'linear.')
        new_state_dict[new_key] = value
    
    # بررسی سازگاری
    model_keys = set(teacher.state_dict().keys())
    checkpoint_keys = set(new_state_dict.keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys or unexpected_keys:
        print(f"⚠ Key mismatch detected:")
        if missing_keys:
            print(f"  Missing keys (first 5): {list(missing_keys)[:5]}")
        if unexpected_keys:
            print(f"  Unexpected keys (first 5): {list(unexpected_keys)[:5]}")
        print("  Attempting to load anyway...")
        teacher.load_state_dict(new_state_dict, strict=False)
        print("✓ Teacher loaded (non-strict)")
    else:
        teacher.load_state_dict(new_state_dict, strict=True)
        print("✓ Teacher loaded successfully (strict)")
    
    return teacher


def parse_args():
    parser = argparse.ArgumentParser(description='PDD Implementation')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--teacher_checkpoint', type=str, 
                        default='checkpoints/resnet56_cifar10.pth')
    parser.add_argument('--download_teacher', action='store_true', default=True,
                        help='Auto-download teacher checkpoint if not found')
    
    # Training (matching paper: 50 epochs for distillation)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--lr_decay_epochs', type=list, default=[20, 40])
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    
    # Distillation
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    
    # Fine-tuning (100 epochs as per paper)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_lr', type=float, default=0.01)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--reg_lambda', type=float, default=0.001) 
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading CIFAR10...")
    train_loader, test_loader = get_cifar10_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    
    # Create models
    print("\nCreating models...")
    student = resnet20(num_classes=10).to(device)
    teacher = resnet56(num_classes=10).to(device)
    
    # Download teacher checkpoint if needed
    if args.download_teacher:
        if not download_teacher_checkpoint(args.teacher_checkpoint):
            print("\n⚠ ERROR: Could not download teacher checkpoint!")
            print("Please download manually from:")
            print("https://github.com/chenyaofo/pytorch-cifar-models")
            return
    
    # Load teacher with automatic key mapping
    print("\nLoading teacher model...")
    teacher = load_teacher_model(teacher, args.teacher_checkpoint, device)
    teacher.eval()
    
    # Evaluate teacher
    print("\nEvaluating teacher model...")
    teacher.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = teacher(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    teacher_acc = 100. * correct / total
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    
    if teacher_acc < 50.0:
        print("\n⚠ WARNING: Teacher accuracy is very low!")
        print("This might indicate an issue with loading the checkpoint.")
        print("Please verify the checkpoint file.")
    
    # Phase 1: Pruning During Distillation
    print("\n" + "="*70)
    print("PHASE 1: Pruning During Distillation (50 epochs)")
    print("="*70)
    
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args)
    trainer.train()
    
    # Save with masks
    save_path = os.path.join(args.save_dir, 'student_with_masks.pth')
    save_checkpoint({
        'state_dict': student.state_dict(),
        'masks': trainer.get_masks(),
        'args': args
    }, save_path)
    print(f"✓ Saved to {save_path}")
    
    # Phase 2: Prune Model
    print("\n" + "="*70)
    print("PHASE 2: Pruning Model")
    print("="*70)
    
    pruner = ModelPruner(student, trainer.get_masks())
    pruned_student = pruner.prune()
    
    orig_params, pruned_params = pruner.get_params_count()
    orig_flops, pruned_flops = pruner.get_flops_count()
    
    params_red = (1 - pruned_params / orig_params) * 100
    flops_red = (1 - pruned_flops / orig_flops) * 100
    
    print(f"\nCompression Results:")
    print(f"Parameters: {orig_params:,} → {pruned_params:,} ({params_red:.2f}% reduction)")
    print(f"FLOPs: {orig_flops:,} → {pruned_flops:,} ({flops_red:.2f}% reduction)")
    
    # Phase 3: Fine-tune
    print("\n" + "="*70)
    print("PHASE 3: Fine-tuning Pruned Model (100 epochs)")
    print("="*70)
    
    pruned_student = pruned_student.to(device)
    
    optimizer = torch.optim.SGD(
        pruned_student.parameters(),
        lr=args.finetune_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 80], gamma=0.1
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.finetune_epochs}] "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': pruned_student.state_dict(),
                'accuracy': test_acc,
                'params_reduction': params_red,
                'flops_reduction': flops_red,
                'args': args
            }, os.path.join(args.save_dir, 'pruned_best.pth'))
        
        scheduler.step()
    
    # Final Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Parameters Reduction: {params_red:.2f}%")
    print(f"FLOPs Reduction: {flops_red:.2f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
