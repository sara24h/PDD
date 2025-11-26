import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
from utils.data_loader_face import Dataset_selector 
from models.resnet import resnet18, resnet50
from utils.trainer import PDDTrainer
from utils.pruner import ModelPruner
from utils.helpers import set_seed, save_checkpoint

def setup_ddp(rank, world_size):
    """تنظیم DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """پاکسازی DDP"""
    dist.destroy_process_group()

def load_teacher_model(teacher, checkpoint_path, device):
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
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '').replace('shortcut.', 'downsample.')
        new_state_dict[new_key] = value
    
    teacher.load_state_dict(new_state_dict, strict=False)
    return teacher

def parse_args():
    parser = argparse.ArgumentParser(description='PDD with DDP for Binary Face Classification')
    
    # DDP
    parser.add_argument('--world_size', type=int, default=2, help='تعداد GPU ها')
    
    # Data
    parser.add_argument('--dataset_mode', type=str, default='rvf10k', 
                       choices=['rvf10k', '140k', '190k', '200k', '330k'])
    parser.add_argument('--rvf10k_train_csv', type=str, default='/kaggle/input/rvf10k/train.csv')
    parser.add_argument('--rvf10k_valid_csv', type=str, default='/kaggle/input/rvf10k/valid.csv')
    parser.add_argument('--rvf10k_root_dir', type=str, default='/kaggle/input/rvf10k')
    parser.add_argument('--realfake140k_train_csv', type=str, default=None)
    parser.add_argument('--realfake140k_valid_csv', type=str, default=None)
    parser.add_argument('--realfake140k_test_csv', type=str, default=None)
    parser.add_argument('--realfake140k_root_dir', type=str, default=None)
    parser.add_argument('--realfake190k_root_dir', type=str, default=None)
    parser.add_argument('--realfake200k_train_csv', type=str, default=None)
    parser.add_argument('--realfake200k_val_csv', type=str, default=None)
    parser.add_argument('--realfake200k_test_csv', type=str, default=None)
    parser.add_argument('--realfake200k_root_dir', type=str, default=None)
    parser.add_argument('--realfake330k_root_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--teacher_checkpoint', type=str, 
                       default='/kaggle/input/10k_teacher_beaet/pytorch/default/1/10k-teacher_model_best.pth')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--lr_decay_epochs', type=list, default=[20, 40])
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    
    # Distillation
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--temperature', '--T', default=4.0, type=float)
    
    # Fine-tuning
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_lr', type=float, default=0.01)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/pdd_checkpoints')
    
    return parser.parse_args()

def main_worker(rank, world_size, args):
    """هر GPU این تابع را اجرا می‌کند"""
    
    # تنظیم DDP
    setup_ddp(rank, world_size)
    set_seed(args.seed + rank)
    
    device = torch.device(f'cuda:{rank}')
    is_main_process = (rank == 0)
    
    if is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Running on {world_size} GPUs with DDP")
        print(f"{'='*70}")
    
    NUM_CLASSES = 1
    
    # بارگذاری دیتاست
    if is_main_process:
        print(f"\nLoading {args.dataset_mode.upper()} Dataset...")
    
    dataset_selector = Dataset_selector(
        dataset_mode=args.dataset_mode,
        rvf10k_train_csv=args.rvf10k_train_csv,
        rvf10k_valid_csv=args.rvf10k_valid_csv,
        rvf10k_root_dir=args.rvf10k_root_dir,
        realfake140k_train_csv=args.realfake140k_train_csv,
        realfake140k_valid_csv=args.realfake140k_valid_csv,
        realfake140k_test_csv=args.realfake140k_test_csv,
        realfake140k_root_dir=args.realfake140k_root_dir,
        realfake190k_root_dir=args.realfake190k_root_dir,
        realfake200k_train_csv=args.realfake200k_train_csv,
        realfake200k_val_csv=args.realfake200k_val_csv,
        realfake200k_test_csv=args.realfake200k_test_csv,
        realfake200k_root_dir=args.realfake200k_root_dir,
        realfake330k_root_dir=args.realfake330k_root_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        ddp=True
    )
    
    train_loader = dataset_selector.loader_train
    test_loader = dataset_selector.loader_test
    
    # ساخت مدل‌ها
    if is_main_process:
        print("\nCreating models...")
    
    student = resnet18(num_classes=NUM_CLASSES).to(device)
    teacher = resnet50(num_classes=NUM_CLASSES).to(device)
    
    # بارگذاری teacher
    if is_main_process:
        print("\nLoading teacher model...")
    
    teacher = load_teacher_model(teacher, args.teacher_checkpoint, device)
    teacher.eval()
    
    # تبدیل به DDP
    student = DDP(student, device_ids=[rank], output_device=rank)
    teacher = DDP(teacher, device_ids=[rank], output_device=rank)
    
    if is_main_process:
        print(f"Student (ResNet18) parameters: {sum(p.numel() for p in student.parameters()):,}")
        print(f"Teacher (ResNet50) parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # ارزیابی teacher
    if is_main_process:
        print("\nEvaluating teacher model...")
        teacher_acc = evaluate_model(teacher, test_loader, device)
        print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
    
    # Phase 1: PDD Training
    if is_main_process:
        print("\n" + "="*70)
        print("PHASE 1: Pruning During Distillation")
        print("="*70)
    
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args, rank, world_size)
    trainer.train()
    
    # فقط main process ذخیره می‌کند
    if is_main_process:
        save_path = os.path.join(args.save_dir, 'student_with_masks.pth')
        save_checkpoint({
            'state_dict': student.module.state_dict(),
            'masks': trainer.get_masks(),
            'args': args
        }, save_path)
        print(f"✓ Saved to {save_path}")
    
    # همگام‌سازی
    dist.barrier()
    
    # Phase 2: Pruning (فقط main process)
    if is_main_process:
        print("\n" + "="*70)
        print("PHASE 2: Pruning Model")
        print("="*70)
        
        pruner = ModelPruner(student.module, trainer.get_masks())
        pruned_student = pruner.prune()
        
        orig_params, pruned_params = pruner.get_params_count()
        orig_flops, pruned_flops = pruner.get_flops_count()
        
        params_red = (1 - pruned_params / orig_params) * 100
        flops_red = (1 - pruned_flops / orig_flops) * 100
        
        print(f"\nCompression Results:")
        print(f"Parameters: {orig_params:,} → {pruned_params:,} ({params_red:.2f}% reduction)")
        print(f"FLOPs: {orig_flops:,} → {pruned_flops:,} ({flops_red:.2f}% reduction)")
        
        # ذخیره مدل هرس شده
        save_checkpoint({
            'state_dict': pruned_student.state_dict(),
            'params_reduction': params_red,
            'flops_reduction': flops_red
        }, os.path.join(args.save_dir, 'pruned_student.pth'))
    
    # همگام‌سازی
    dist.barrier()
    
    # Phase 3: Fine-tuning
    if is_main_process:
        print("\n" + "="*70)
        print("PHASE 3: Fine-tuning Pruned Model")
        print("="*70)
    
    # بارگذاری مدل هرس شده
    checkpoint = torch.load(os.path.join(args.save_dir, 'pruned_student.pth'), map_location=device)
    
    # ساخت مجدد مدل هرس شده
    from models.resnet import ResNet, BasicBlock
    pruned_student = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES)
    pruned_student.load_state_dict(checkpoint['state_dict'])
    pruned_student = pruned_student.to(device)
    pruned_student = DDP(pruned_student, device_ids=[rank], output_device=rank)
    
    # Fine-tuning
    finetune_model(pruned_student, train_loader, test_loader, device, args, rank, is_main_process)
    
    cleanup_ddp()

def evaluate_model(model, test_loader, device):
    """ارزیابی مدل"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(1)
            preds = (outputs > 0).long()
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    
    return 100. * correct / total

def finetune_model(model, train_loader, test_loader, device, args, rank, is_main_process):
    """Fine-tuning مدل هرس شده"""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.finetune_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 80], gamma=0.1
    )
    
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    
    from tqdm import tqdm
    import time
    
    for epoch in range(args.finetune_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        
        correct = 0
        total = 0
        epoch_loss = 0.0
        
        if is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.finetune_epochs}")
            start_time = time.time()
        else:
            pbar = train_loader
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            preds = (outputs > 0).long()
            total += targets.size(0)
            correct += preds.eq(targets.long()).sum().item()
            epoch_loss += loss.item()
            
            if is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        train_acc = 100. * correct / total
        
        # ارزیابی
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze(1)
                preds = (outputs > 0).long()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        
        if is_main_process:
            elapsed = time.time() - start_time
            print(f"\nEpoch [{epoch+1}/{args.finetune_epochs}] - Time: {elapsed:.1f}s")
            print(f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'accuracy': test_acc,
                    'args': args
                }, os.path.join(args.save_dir, 'pruned_finetuned_best.pth'))
                print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        scheduler.step()

def main():
    args = parse_args()
    
    # بررسی تعداد GPU
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    world_size = min(args.world_size, torch.cuda.device_count())
    print(f"Using {world_size} GPUs")
    
    # اجرای DDP
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
