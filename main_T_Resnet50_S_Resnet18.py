import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import time
from tqdm import tqdm
from utils.data_loader_face import Dataset_selector 
from models.resnet import resnet18, resnet50
from utils.trainer import PDDTrainer
from utils.pruner import ModelPruner
from utils.helpers import set_seed, save_checkpoint

def setup_ddp(rank, world_size):
    """Initialize DDP environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleanup DDP"""
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
    
    model_keys = set(teacher.state_dict().keys())
    checkpoint_keys = set(new_state_dict.keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys or unexpected_keys:
        if dist.get_rank() == 0:
            print(f"⚠ Key mismatch detected:")
            if missing_keys:
                print(f"  Missing keys (first 5): {list(missing_keys)[:5]}")
            if unexpected_keys:
                print(f"  Unexpected keys (first 5): {list(unexpected_keys)[:5]}")
            print("  Attempting to load anyway...")
        teacher.load_state_dict(new_state_dict, strict=False)
        if dist.get_rank() == 0:
            print("✓ Teacher loaded (non-strict)")
    else:
        teacher.load_state_dict(new_state_dict, strict=True)
        if dist.get_rank() == 0:
            print("✓ Teacher loaded successfully (strict)")
    
    return teacher

def parse_args():
    parser = argparse.ArgumentParser(description='PDD for Binary Face Classification with DDP')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='rvf10k',
                       choices=['rvf10k', '140k', '190k', '200k', '330k'],
                       help='Dataset to use')
    
    # RVF10K paths
    parser.add_argument('--rvf10k_train_csv', type=str, default='/kaggle/input/rvf10k/train.csv')
    parser.add_argument('--rvf10k_valid_csv', type=str, default='/kaggle/input/rvf10k/valid.csv')
    parser.add_argument('--rvf10k_root_dir', type=str, default='/kaggle/input/rvf10k')
    
    # 140K paths
    parser.add_argument('--realfake140k_train_csv', type=str, 
                       default='/kaggle/input/140k-real-and-fake-faces/train.csv')
    parser.add_argument('--realfake140k_valid_csv', type=str,
                       default='/kaggle/input/140k-real-and-fake-faces/valid.csv')
    parser.add_argument('--realfake140k_test_csv', type=str,
                       default='/kaggle/input/140k-real-and-fake-faces/test.csv')
    parser.add_argument('--realfake140k_root_dir', type=str,
                       default='/kaggle/input/140k-real-and-fake-faces')
    
    # 190K paths
    parser.add_argument('--realfake190k_root_dir', type=str,
                       default='/kaggle/input/deepfake-and-real-images/Dataset')
    
    # 200K paths
    parser.add_argument('--realfake200k_train_csv', type=str,
                       default='/kaggle/input/200k-real-and-fake-faces/train_labels.csv')
    parser.add_argument('--realfake200k_val_csv', type=str,
                       default='/kaggle/input/200k-real-and-fake-faces/val_labels.csv')
    parser.add_argument('--realfake200k_test_csv', type=str,
                       default='/kaggle/input/200k-real-and-fake-faces/test_labels.csv')
    parser.add_argument('--realfake200k_root_dir', type=str,
                       default='/kaggle/input/200k-real-and-fake-faces')
    
    # 330K paths
    parser.add_argument('--realfake330k_root_dir', type=str,
                       default='/kaggle/input/deepfake-dataset')
    
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
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save periodic checkpoints') 
    parser.add_argument('--resume_path', type=str, default=None, help='Path to a checkpoint to resume training from') # --- پایان آرگومان‌های جدید --- # مسیر ذخیره چک‌پوینت نهایی PDD 
    parser.add_argument('--pdd_checkpoint_path', type=str, default='./pdd_checkpoint.pth', help='Path to save the final PDD checkpoint')
    # DDP
    parser.add_argument('--world_size', type=int, default=2, help='Number of GPUs')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/pdd_checkpoints')
    
    return parser.parse_args()

def main_worker(rank, world_size, args):
    """Main training function for each process"""
    
    # Setup DDP
    setup_ddp(rank, world_size)
    set_seed(args.seed + rank)
    
    device = torch.device(f'cuda:{rank}')
    is_main = (rank == 0)
    
    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Training on {world_size} GPUs with DDP")
        print(f"Dataset: {args.dataset}")
        print(f"{'='*70}\n")
    
    NUM_CLASSES = 1
    
    # Load data with DDP
    if is_main:
        print(f"\nLoading {args.dataset} Dataset...")
    
    dataset_kwargs = {
        'dataset_mode': args.dataset,
        'train_batch_size': args.batch_size,
        'eval_batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'ddp': True
    }
    
    # Add dataset-specific paths
    if args.dataset == 'rvf10k':
        dataset_kwargs.update({
            'rvf10k_train_csv': args.rvf10k_train_csv,
            'rvf10k_valid_csv': args.rvf10k_valid_csv,
            'rvf10k_root_dir': args.rvf10k_root_dir,
        })
    elif args.dataset == '140k':
        dataset_kwargs.update({
            'realfake140k_train_csv': args.realfake140k_train_csv,
            'realfake140k_valid_csv': args.realfake140k_valid_csv,
            'realfake140k_test_csv': args.realfake140k_test_csv,
            'realfake140k_root_dir': args.realfake140k_root_dir,
        })
    elif args.dataset == '190k':
        dataset_kwargs.update({
            'realfake190k_root_dir': args.realfake190k_root_dir,
        })
    elif args.dataset == '200k':
        dataset_kwargs.update({
            'realfake200k_train_csv': args.realfake200k_train_csv,
            'realfake200k_val_csv': args.realfake200k_val_csv,
            'realfake200k_test_csv': args.realfake200k_test_csv,
            'realfake200k_root_dir': args.realfake200k_root_dir,
        })
    elif args.dataset == '330k':
        dataset_kwargs.update({
            'realfake330k_root_dir': args.realfake330k_root_dir,
        })
    
    dataset_selector = Dataset_selector(**dataset_kwargs)
    train_loader = dataset_selector.loader_train
    test_loader = dataset_selector.loader_test
    
    # Create models
    if is_main:
        print("\nCreating models...")
    
    student = resnet18(num_classes=NUM_CLASSES).to(device)
    teacher = resnet50(num_classes=NUM_CLASSES).to(device)
    
    # Wrap with DDP
    student = DDP(student, device_ids=[rank])
    
    if is_main:
        print(f"Student (ResNet18) parameters: {sum(p.numel() for p in student.parameters()):,}")
        print(f"Teacher (ResNet50) parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # Load teacher
    if is_main:
        print("\nLoading teacher model...")
    
    if not os.path.exists(args.teacher_checkpoint):
        if is_main:
            print(f"✗ ERROR: Teacher checkpoint not found at {args.teacher_checkpoint}")
        cleanup_ddp()
        return
    
    teacher = load_teacher_model(teacher, args.teacher_checkpoint, device)
    teacher.eval()
    
    # Evaluate teacher
    if is_main:
        print("\nEvaluating teacher model...")
    
    teacher_acc = evaluate_model(teacher, test_loader, device, rank, world_size)
    
    if is_main:
        print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
    
    # Phase 1: PDD Training
    if is_main:
        print("\n" + "="*70)
        print("PHASE 1: Pruning During Distillation")
        print("="*70)
    
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args, rank, world_size)
    trainer.train()
    
    # Save checkpoint (only rank 0)
    if is_main:
        save_path = os.path.join(args.save_dir, f'student_resnet18_{args.dataset}_with_masks.pth')
        save_checkpoint({
            'state_dict': student.module.state_dict(),
            'masks': trainer.get_masks(),
            'args': args
        }, save_path)
        print(f"✓ Saved to {save_path}")
    
    dist.barrier()
    
    # Phase 2: Prune (only on rank 0)
    if is_main:
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
    else:
        pruned_student = None
        params_red = None
        flops_red = None
    
    dist.barrier()
    
    # Broadcast pruned model to all ranks
    if not is_main:
        student_copy = resnet18(num_classes=NUM_CLASSES)
        pruner = ModelPruner(student_copy, trainer.get_masks())
        pruned_student = pruner.prune()
        params_red = 0
        flops_red = 0
    
    # Phase 3: Fine-tuning with DDP
    if is_main:
        print("\n" + "="*70)
        print("PHASE 3: Fine-tuning Pruned Model")
        print("="*70)
    
    pruned_student = pruned_student.to(device)
    pruned_student = DDP(pruned_student, device_ids=[rank])
    
    best_acc = finetune_model(pruned_student, train_loader, test_loader, device, args, rank, world_size, is_main)
    
    # Save final model
    if is_main:
        save_checkpoint({
            'state_dict': pruned_student.module.state_dict(),
            'accuracy': best_acc,
            'params_reduction': params_red,
            'flops_reduction': flops_red,
            'args': args
        }, os.path.join(args.save_dir, f'pruned_resnet18_{args.dataset}_best.pth'))
        
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
        print(f"Best Test Accuracy (Pruned ResNet18): {best_acc:.2f}%")
        print(f"Parameters Reduction: {params_red:.2f}%")
        print(f"FLOPs Reduction: {flops_red:.2f}%")
        print("="*70 + "\n")
    
    cleanup_ddp()

def evaluate_model(model, test_loader, device, rank, world_size):
    """Evaluate model across all GPUs"""
    model.eval()
    correct = torch.tensor(0.0).to(device)
    total = torch.tensor(0.0).to(device)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(1)
            preds = (outputs > 0).long()
            correct += preds.eq(targets).sum()
            total += targets.size(0)
    
    # Aggregate across GPUs
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    
    return 100. * correct.item() / total.item()

def finetune_model(model, train_loader, test_loader, device, args, rank, world_size, is_main):
    """Fine-tune pruned model with progress bar"""
    
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
    
    start_time = time.time()
    
    for epoch in range(args.finetune_epochs):
        epoch_start = time.time()
        
        model.train()
        train_loader.sampler.set_epoch(epoch)
        
        correct = torch.tensor(0.0).to(device)
        total = torch.tensor(0.0).to(device)
        running_loss = torch.tensor(0.0).to(device)
        
        # Progress bar only on main process
        if is_main:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.finetune_epochs}")
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
            correct += preds.eq(targets.long()).sum()
            running_loss += loss.item()
            
            if is_main:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Aggregate training stats
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        
        train_acc = 100. * correct.item() / total.item()
        avg_loss = running_loss.item() / len(train_loader)
        
        # Evaluate
        test_acc = evaluate_model(model, test_loader, device, rank, world_size)
        
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        
        if is_main:
            print(f"Epoch [{epoch+1}/{args.finetune_epochs}] "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
                  f"Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | "
                  f"Elapsed: {elapsed_time/60:.1f}min")
        
        if test_acc > best_acc:
            best_acc = test_acc
            if is_main:
                print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        scheduler.step()
    
    return best_acc

def main():
    args = parse_args()
    
    # Check if GPUs are available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    if torch.cuda.device_count() < args.world_size:
        print(f"ERROR: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available")
        return
    
    # Start multiprocessing
    mp.spawn(
        main_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == '__main__':
    main()
