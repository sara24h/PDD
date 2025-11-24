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
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace('module.', '')
        new_key = new_key.replace('downsample.', 'shortcut.')
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
    # <<< CHANGE: توضیحات را برای مسئله خودتان تغییر دهید
    parser = argparse.ArgumentParser(description='PDD for Binary Face Classification (RVF10K)')
    
    # Data
    # <<< CHANGE: آرگومان‌های دیتاست RVF10K را اضافه کنید
    parser.add_argument('--rvf10k_train_csv', type=str, default='/kaggle/input/rvf10k/train.csv')
    parser.add_argument('--rvf10k_valid_csv', type=str, default='/kaggle/input/rvf10k/valid.csv')
    parser.add_argument('--rvf10k_root_dir', type=str, default='/kaggle/input/rvf10k')
    parser.add_argument('--batch_size', type=int, default=64) # <<< CHANGE: کاهش batch size برای دیتاست بزرگتر
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    # <<< CHANGE: مسیر پیش‌فرض را به معلم خودتان تغییر دهید
    parser.add_argument('--teacher_checkpoint', type=str, 
                        default='/kaggle/input/10k_teacher_beaet/pytorch/default/1/10k-teacher_model_best.pth')
    
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
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/pdd_checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # <<< CHANGE: تعداد کلاس‌ها را 2 تعریف کنید
    NUM_CLASSES = 2
    
    print(f"Device: {device}")
    print(f"Task: Binary Face Classification (RVF10K)")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Student Model: ResNet18")
    print(f"Teacher Model: ResNet50")
    
    # <<< CHANGE: از لودر داده سفارشی خود برای RVF10K استفاده کنید
    print("\nLoading RVF10K Dataset...")
    dataset_selector = Dataset_selector(
        dataset_mode='rvf10k',
        rvf10k_train_csv=args.rvf10k_train_csv,
        rvf10k_valid_csv=args.rvf10k_valid_csv,
        rvf10k_root_dir=args.rvf10k_root_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        ddp=False # فرض بر استفاده از یک GPU
    )
    train_loader = dataset_selector.loader_train
    test_loader = dataset_selector.loader_test # استفاده از test_loader برای ارزیابی
    
    # Create models
    print("\nCreating models...")
    # <<< CHANGE: از ResNet18 به عنوان دانش‌آموز و با 2 کلاس استفاده کنید
    student = resnet18(num_classes=NUM_CLASSES).to(device)
    # <<< CHANGE: از ResNet50 به عنوان معلم و با 2 کلاس استفاده کنید
    teacher = resnet50(num_classes=NUM_CLASSES).to(device)
    
    print(f"Student (ResNet18) parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"Teacher (ResNet50) parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # <<< CHANGE: بخش دانلود معلم را کاملاً حذف کنید
    
    # Load teacher with automatic key mapping
    print("\nLoading teacher model...")
    # بررسی کنید که فایل معلم وجود دارد
    if not os.path.exists(args.teacher_checkpoint):
        print(f"✗ ERROR: Teacher checkpoint not found at {args.teacher_checkpoint}")
        print("Please check the path to your teacher model.")
        return
        
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
    print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
    
    if teacher_acc < 70.0: # برای باینری، انتظار دقت بالاتری داریم
        print("\n⚠ WARNING: Teacher accuracy might be too low for effective distillation!")
    
    # Phase 1: Pruning During Distillation
    print("\n" + "="*70)
    print("PHASE 1: Pruning During Distillation (50 epochs)")
    print("="*70)
    
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args)
    trainer.train()
    
    # <<< CHANGE: نام فایل ذخیره‌سازی را تغییر دهید
    save_path = os.path.join(args.save_dir, 'student_resnet18_binary_with_masks.pth')
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
        
        
        print(f"Epoch [{epoch+1}/{args.finetune_epochs}] "
            f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # <<< CHANGE: نام فایل نهایی را تغییر دهید
            save_checkpoint({
                'epoch': epoch,
                'state_dict': pruned_student.state_dict(),
                'accuracy': test_acc,
                'params_reduction': params_red,
                'flops_reduction': flops_red,
                'args': args
            }, os.path.join(args.save_dir, 'pruned_resnet18_binary_best.pth'))
        
        scheduler.step()
    
    # Final Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
    print(f"Best Test Accuracy (Pruned ResNet18): {best_acc:.2f}%")
    print(f"Parameters Reduction: {params_red:.2f}%")
    print(f"FLOPs Reduction: {flops_red:.2f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
