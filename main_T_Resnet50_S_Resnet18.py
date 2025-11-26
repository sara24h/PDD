import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import os
import time
from tqdm import tqdm
from utils.data_loader_face import Dataset_selector 
from models.resnet import resnet18, resnet50
from utils.trainer import PDDTrainer
from utils.pruner import ModelPruner
from utils.helpers import set_seed, save_checkpoint

def setup(rank, world_size):
    """Initialize the distributed process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed process group"""
    dist.destroy_process_group()

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
    
    # DDP parameters
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                        help='Number of processes for DDP')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    return parser.parse_args()


def load_dataset(args, rank):
    """Load dataset based on args.dataset with DistributedSampler"""
    
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
            ddp=True,
            rank=rank,
            world_size=args.world_size
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
            ddp=True,
            rank=rank,
            world_size=args.world_size
        )
    
    elif args.dataset == '190k':
        dataset_selector = Dataset_selector(
            dataset_mode='190k',
            realfake190k_root_dir=args.realfake190k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=True,
            rank=rank,
            world_size=args.world_size
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
            ddp=True,
            rank=rank,
            world_size=args.world_size
        )
    
    elif args.dataset == '330k':
        dataset_selector = Dataset_selector(
            dataset_mode='330k',
            realfake330k_root_dir=args.realfake330k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            ddp=True,
            rank=rank,
            world_size=args.world_size
        )
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"✓ Dataset loaded successfully\n")
    
    return dataset_selector.loader_train, dataset_selector.loader_test


def main_worker(rank, args):
    # Setup DDP
    setup(rank, args.world_size)
    
    # Set seed
    set_seed(args.seed + rank)
    
    # Create save directory
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{rank}')
    
    # Binary classification: 1 output
    NUM_CLASSES = 1
    
    # Print configuration (only on rank 0)
    if rank == 0:
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
        print(f"World Size: {args.world_size}")
        print("="*70)
    
    # Load data
    train_loader, test_loader = load_dataset(args, rank)
    
    # Create models
    print(f"\n[Rank {rank}] Creating models...")
    student = resnet18(num_classes=NUM_CLASSES).to(device)
    teacher = resnet50(num_classes=NUM_CLASSES).to(device)
    
    # Wrap models with DDP
    student = DDP(student, device_ids=[rank])
    teacher = DDP(teacher, device_ids=[rank])
    
    if rank == 0:
        print(f"Student (ResNet18) parameters: {sum(p.numel() for p in student.parameters()):,}")
        print(f"Teacher (ResNet50) parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # Load teacher (only on rank 0, then broadcast)
    if rank == 0:
        print("\nLoading teacher model...")
        if not os.path.exists(args.teacher_checkpoint):
            print(f"✗ ERROR: Teacher checkpoint not found at {args.teacher_checkpoint}")
            cleanup()
            return
        
        teacher_state = load_teacher_model(teacher.module, args.teacher_checkpoint, device).state_dict()
    else:
        teacher_state = None
    
    # Broadcast teacher state to all processes
    if args.world_size > 1:
        dist.barrier()
    
    if rank == 0:
        # Save teacher state to a temporary file
        temp_path = os.path.join(args.save_dir, 'temp_teacher.pth')
        torch.save(teacher_state, temp_path)
        dist.barrier()
    else:
        # Wait for rank 0 to save the file
        dist.barrier()
        # Load teacher state from the temporary file
        temp_path = os.path.join(args.save_dir, 'temp_teacher.pth')
        teacher_state = torch.load(temp_path, map_location=device)
        dist.barrier()
    
    # Load teacher state on all processes
    teacher.module.load_state_dict(teacher_state)
    teacher.eval()
    
    # Evaluate teacher (only on rank 0)
    if rank == 0:
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
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 1: Pruning During Distillation")
        print("="*70)
    
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args, rank=rank)
    
    # Track training time for distillation
    if rank == 0:
        distillation_start_time = time.time()
    
    trainer.train()
    
    # Calculate and print distillation time
    if rank == 0:
        distillation_time = time.time() - distillation_start_time
        print(f"\nDistillation completed in {distillation_time/60:.2f} minutes")
    
    # Save student with masks (only on rank 0)
    if rank == 0:
        checkpoint_name = f'student_resnet18_{args.dataset}_with_masks.pth'
        save_path = os.path.join(args.save_dir, checkpoint_name)
        save_checkpoint({
            'state_dict': student.module.state_dict(),
            'masks': trainer.get_masks(),
            'best_acc': trainer.best_acc,
            'args': vars(args),
            'dataset': args.dataset
        }, save_path)
        print(f"✓ Saved checkpoint to {save_path}")
    
    # Phase 2: Prune (only on rank 0)
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 2: Pruning Model")
        print("="*70)
        
        pruner = ModelPruner(student.module, trainer.get_masks())
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
        
        # Save pruned model to a temporary file for all processes to load
        temp_pruned_path = os.path.join(args.save_dir, 'temp_pruned.pth')
        torch.save(pruned_student.state_dict(), temp_pruned_path)
    
    # Synchronize all processes
    if args.world_size > 1:
        dist.barrier()
    
    # Load pruned model on all processes
    if rank != 0:
        temp_pruned_path = os.path.join(args.save_dir, 'temp_pruned.pth')
        pruned_student = resnet18(num_classes=NUM_CLASSES).to(device)
        pruned_student.load_state_dict(torch.load(temp_pruned_path, map_location=device))
    
    # Wrap pruned model with DDP
    pruned_student = DDP(pruned_student, device_ids=[rank])
    
    # Phase 3: Fine-tune
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 3: Fine-tuning Pruned Model")
        print("="*70)
    
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
    
    # Track fine-tuning time
    if rank == 0:
        finetune_start_time = time.time()
        total_epochs = args.finetune_epochs
        epoch_progress = tqdm(range(total_epochs), desc="Fine-tuning Progress", position=0, leave=True)
    
    for epoch in range(args.finetune_epochs):
        # Training
        pruned_student.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Track epoch time
        if rank == 0:
            epoch_start_time = time.time()
            batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", 
                                 leave=False, position=1)
        
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
            
            # Update batch progress bar
            if rank == 0:
                batch_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        if rank == 0:
            batch_progress.close()
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Evaluation
        pruned_student.eval()
        correct = 0
        total = 0
        
        if rank == 0:
            eval_progress = tqdm(test_loader, desc="Evaluating", leave=False, position=2)
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pruned_student(inputs).squeeze(1)
                preds = (outputs > 0).long()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
                
                # Update evaluation progress bar
                if rank == 0:
                    eval_progress.set_postfix({
                        'Acc': f'{100.*correct/total:.2f}%'
                    })
        
        if rank == 0:
            eval_progress.close()
        
        test_acc = 100. * correct / total
        
        # Calculate epoch time
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            remaining_time = epoch_time * (total_epochs - epoch - 1)
            
            # Update epoch progress bar
            epoch_progress.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.2f}%',
                'Test Acc': f'{test_acc:.2f}%',
                'Epoch Time': f'{epoch_time:.2f}s',
                'ETA': f'{remaining_time/60:.1f}m'
            })
        
        # Save best model (only on rank 0)
        if rank == 0 and test_acc > best_acc:
            best_acc = test_acc
            best_checkpoint_name = f'pruned_resnet18_{args.dataset}_best.pth'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': pruned_student.module.state_dict(),
                'accuracy': test_acc,
                'params_reduction': params_red,
                'flops_reduction': flops_red,
                'args': vars(args),
                'dataset': args.dataset
            }, os.path.join(args.save_dir, best_checkpoint_name))
        
        scheduler.step()
    
    if rank == 0:
        epoch_progress.close()
        finetune_time = time.time() - finetune_start_time
        print(f"\nFine-tuning completed in {finetune_time/60:.2f} minutes")
    
    # Final Results (only on rank 0)
    if rank == 0:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
        print(f"Student (ResNet18) Best Accuracy after Distillation: {trainer.best_acc:.2f}%")
        print(f"Pruned Student Best Accuracy after Fine-tuning: {best_acc:.2f}%")
        print(f"Parameters Reduction: {params_red:.2f}%")
        print(f"FLOPs Reduction: {flops_red:.2f}%")
        print(f"Distillation Time: {distillation_time/60:.2f} minutes")
        print(f"Fine-tuning Time: {finetune_time/60:.2f} minutes")
        print(f"Total Training Time: {(distillation_time+finetune_time)/60:.2f} minutes")
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
            'pruned_flops': pruned_flops,
            'distillation_time_minutes': distillation_time/60,
            'finetune_time_minutes': finetune_time/60,
            'total_time_minutes': (distillation_time+finetune_time)/60
        }
        
        import json
        summary_path = os.path.join(args.save_dir, f'summary_{args.dataset}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✓ Summary saved to {summary_path}\n")

    cleanup()


def main():
    args = parse_args()

    if args.world_size > 1:
        mp.spawn(main_worker,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)
    else:
        main_worker(0, args)


if __name__ == '__main__':
    main()
