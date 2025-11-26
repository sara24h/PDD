import torch
import torch.nn as nn
import argparse
import os
from utils.data_loader_face import Dataset_selector 
from models.resnet import resnet18, resnet50
from utils.trainer import PDDTrainer
from utils.pruner import ModelPruner
from utils.helpers import set_seed, save_checkpoint

def load_teacher_model(teacher, checkpoint_path, device):
    """Load teacher model with flexible key matching"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    # Clean up keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '').replace('shortcut.', 'downsample.')
        new_state_dict[new_key] = value
    
    # Check key compatibility
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
    parser = argparse.ArgumentParser(
        description='PDD for Binary Face Classification - Multi-Dataset Support'
    )
    
    # ==========================================
    # Dataset Selection
    # ==========================================
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['rvf10k', '140k', '190k', '200k', '330k'],
        default='rvf10k',
        help='Choose dataset: rvf10k, 140k, 190k, 200k, or 330k'
    )
    
    # ==========================================
    # RVF10K Dataset Paths
    # ==========================================
    parser.add_argument('--rvf10k_train_csv', type=str, 
                        default='/kaggle/input/rvf10k/train.csv')
    parser.add_argument('--rvf10k_valid_csv', type=str, 
                        default='/kaggle/input/rvf10k/valid.csv')
    parser.add_argument('--rvf10k_root_dir', type=str, 
                        default='/kaggle/input/rvf10k')
    
    # ==========================================
    # 140K Dataset Paths
    # ==========================================
    parser.add_argument('--realfake140k_train_csv', type=str,
                        default='/kaggle/input/140k-real-and-fake-faces/train.csv')
    parser.add_argument('--realfake140k_valid_csv', type=str,
                        default='/kaggle/input/140k-real-and-fake-faces/valid.csv')
    parser.add_argument('--realfake140k_test_csv', type=str,
                        default='/kaggle/input/140k-real-and-fake-faces/test.csv')
    parser.add_argument('--realfake140k_root_dir', type=str,
                        default='/kaggle/input/140k-real-and-fake-faces')
    
    # ==========================================
    # 190K Dataset Paths
    # ==========================================
    parser.add_argument('--realfake190k_root_dir', type=str,
                        default='/kaggle/input/deepfake-and-real-images/Dataset')
    
    # ==========================================
    # 200K Dataset Paths
    # ==========================================
    parser.add_argument('--realfake200k_train_csv', type=str,
                        default='/kaggle/input/200k-real-and-fake-faces/train_labels.csv')
    parser.add_argument('--realfake200k_val_csv', type=str,
                        default='/kaggle/input/200k-real-and-fake-faces/val_labels.csv')
    parser.add_argument('--realfake200k_test_csv', type=str,
                        default='/kaggle/input/200k-real-and-fake-faces/test_labels.csv')
    parser.add_argument('--realfake200k_root_dir', type=str,
                        default='/kaggle/input/200k-real-and-fake-faces')
    
    # ==========================================
    # 330K Dataset Paths
    # ==========================================
    parser.add_argument('--realfake330k_root_dir', type=str,
                        default='/kaggle/input/deepfake-dataset')
    
    # ==========================================
    # Training Hyperparameters
    # ==========================================
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # ==========================================
    # Model
    # ==========================================
    parser.add_argument('--teacher_checkpoint', type=str, required=True,
                        help='Path to pretrained teacher model')
    
    # ==========================================
    # Training
    # ==========================================
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of distillation epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[20, 40],
                        help='Epochs to decay learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    
    # ==========================================
    # Distillation
    # ==========================================
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='Weight for KD loss (CE weight = 1-alpha)')
    parser.add_argument('--temperature', '-T', type=float, default=4.0,
                        help='Temperature for Knowledge Distillation')
    
    # ==========================================
    # Fine-tuning
    # ==========================================
    parser.add_argument('--finetune_epochs', type=int, default=100,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--finetune_lr', type=float, default=0.01,
                        help='Learning rate for fine-tuning')
    
    # ==========================================
    # Other
    # ==========================================
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/pdd_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    return parser.parse_args()


def load_dataset(args):
    """Load dataset based on args.dataset"""
    
    print(f"\n{'='*70}")
    print(f"Loading Dataset: {args.dataset.upper()}")
    print(f"{'='*70}")
    
    if args.dataset == 'rvf10k':
        dataset_selector = Dataset_selector(
            dataset_mode='rvf10k',
            rvf10k_train_csv=args.rvf10k_train_csv,
            rvf10k_valid_csv=args.rvf10k_valid_csv,
            rvf10k_root_dir=args.rvf10k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=False
        )
    
    elif args.dataset == '140k':
        dataset_selector = Dataset_selector(
            dataset_mode='140k',
            realfake140k_train_csv=args.realfake140k_train_csv,
            realfake140k_valid_csv=args.realfake140k_valid_csv,
            realfake140k_test_csv=args.realfake140k_test_csv,
            realfake140k_root_dir=args.realfake140k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=False
        )
    
    elif args.dataset == '190k':
        dataset_selector = Dataset_selector(
            dataset_mode='190k',
            realfake190k_root_dir=args.realfake190k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=False
        )
    
    elif args.dataset == '200k':
        dataset_selector = Dataset_selector(
            dataset_mode='200k',
            realfake200k_train_csv=args.realfake200k_train_csv,
            realfake200k_val_csv=args.realfake200k_val_csv,
            realfake200k_test_csv=args.realfake200k_test_csv,
            realfake200k_root_dir=args.realfake200k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=False
        )
    
    elif args.dataset == '330k':
        dataset_selector = Dataset_selector(
            dataset_mode='330k',
            realfake330k_root_dir=args.realfake330k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=False
        )
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"✓ Dataset loaded successfully\n")
    
    return dataset_selector.loader_train, dataset_selector.loader_test


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Binary classification: 1 output
    NUM_CLASSES = 1
    
    # Print configuration
    print("\n" + "="*70)
    print("PDD: Pruning During Distillation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Task: Binary Face Classification (Real vs Fake)")
    print(f"Student Model: ResNet18 (1 output)")
    print(f"Teacher Model: ResNet50 (1 output)")
    print(f"Batch Size: {args.batch_size}")
    print(f"Distillation Epochs: {args.epochs}")
    print(f"Fine-tuning Epochs: {args.finetune_epochs}")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha (KD weight): {args.alpha}")
    print("="*70)
    
    # Load data
    train_loader, test_loader = load_dataset(args)
    
    # Create models
    print("\nCreating models...")
    student = resnet18(num_classes=NUM_CLASSES).to(device)
    teacher = resnet50(num_classes=NUM_CLASSES).to(device)
    
    print(f"Student (ResNet18) parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Teacher (ResNet50) parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # Load teacher
    print("\nLoading teacher model...")
    if not os.path.exists(args.teacher_checkpoint):
        print(f"✗ ERROR: Teacher checkpoint not found at {args.teacher_checkpoint}")
        return
    
    teacher = load_teacher_model(teacher, args.teacher_checkpoint, device)
    teacher.eval()
    
    # Evaluate teacher
    print("\nEvaluating teacher model...")
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = teacher(inputs).squeeze(1)  # [B]
            preds = (outputs > 0).long()
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    
    teacher_acc = 100. * correct / total
    print(f"Teacher (ResNet50) Test Accuracy: {teacher_acc:.2f}%")
    
    # Phase 1: PDD (Pruning During Distillation)
    print("\n" + "="*70)
    print("PHASE 1: Pruning During Distillation")
    print("="*70)
    
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args)
    trainer.train()
    
    # Save student with masks
    checkpoint_name = f'student_resnet18_{args.dataset}_with_masks.pth'
    save_path = os.path.join(args.save_dir, checkpoint_name)
    save_checkpoint({
        'state_dict': student.state_dict(),
        'masks': trainer.get_masks(),
        'best_acc': trainer.best_acc,
        'args': vars(args),
        'dataset': args.dataset
    }, save_path)
    print(f"✓ Saved checkpoint to {save_path}")
    
    # Phase 2: Prune
    print("\n" + "="*70)
    print("PHASE 2: Pruning Model")
    print("="*70)
    
    pruner = ModelPruner(student, trainer.get_masks())
    pruned_student = pruner.prune()
    
    orig_params, pruned_params = pruner.get_params_count()
    orig_flops, pruned_flops = pruner.get_flops_count()
    
    params_red = (1 - pruned_params / orig_params) * 100
    flops_red = (1 - pruned_flops / orig_flops) * 100
    
    print(f"\n{'='*70}")
    print("Compression Statistics:")
    print(f"{'='*70}")
    print(f"Parameters: {orig_params:,} → {pruned_params:,} ({params_red:.2f}% reduction)")
    print(f"FLOPs: {orig_flops:,} → {pruned_flops:,} ({flops_red:.2f}% reduction)")
    print(f"{'='*70}\n")
    
    # Phase 3: Fine-tune
    print("\n" + "="*70)
    print("PHASE 3: Fine-tuning Pruned Model")
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
    
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    
    for epoch in range(args.finetune_epochs):
        # Training
        pruned_student.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.float()
            
            optimizer.zero_grad()
            outputs = pruned_student(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs > 0).long()
            total += targets.size(0)
            correct += preds.eq(targets.long()).sum().item()
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Evaluation
        pruned_student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pruned_student(inputs).squeeze(1)
                preds = (outputs > 0).long()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{args.finetune_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_checkpoint_name = f'pruned_resnet18_{args.dataset}_best.pth'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': pruned_student.state_dict(),
                'accuracy': test_acc,
                'params_reduction': params_red,
                'flops_reduction': flops_red,
                'args': vars(args),
                'dataset': args.dataset
            }, os.path.join(args.save_dir, best_checkpoint_name))
        
        scheduler.step()
    
    # Final Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
    print(f"Student (ResNet18) Best Accuracy after Distillation: {trainer.best_acc:.2f}%")
    print(f"Pruned Student Best Accuracy after Fine-tuning: {best_acc:.2f}%")
    print(f"Parameters Reduction: {params_red:.2f}%")
    print(f"FLOPs Reduction: {flops_red:.2f}%")
    print("="*70 + "\n")
    
    # Save final summary
    summary = {
        'dataset': args.dataset,
        'teacher_acc': teacher_acc,
        'student_after_distillation': trainer.best_acc,
        'pruned_student_after_finetune': best_acc,
        'params_reduction': params_red,
        'flops_reduction': flops_red,
        'original_params': orig_params,
        'pruned_params': pruned_params,
        'original_flops': orig_flops,
        'pruned_flops': pruned_flops
    }
    
    import json
    summary_path = os.path.join(args.save_dir, f'summary_{args.dataset}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✓ Summary saved to {summary_path}\n")


if __name__ == '__main__':
    main()
