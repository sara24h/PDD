import torch
import torch.nn as nn
import torch.distributed as dist
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
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up the distributed environment."""
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
    
    # DDP settings
    parser.add_argument('--use_ddp', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs to use')
    
    # Dataset Selection
    parser.add_argument('--dataset_mode', type=str, default='rvf10k',
                        choices=['rvf10k', '140k', '190k', '200k', '330k'],
                        help='Select dataset: rvf10k, 140k, 190k, 200k, or 330k')
    
    # RVF10K Dataset
    parser.add_argument('--rvf10k_train_csv', type=str, default='/kaggle/input/rvf10k/train.csv')
    parser.add_argument('--rvf10k_valid_csv', type=str, default='/kaggle/input/rvf10k/valid.csv')
    parser.add_argument('--rvf10k_root_dir', type=str, default='/kaggle/input/rvf10k')
    
    # 140K Dataset
    parser.add_argument('--realfake140k_train_csv', type=str, default='/kaggle/input/140k-real-and-fake-faces/train.csv')
    parser.add_argument('--realfake140k_valid_csv', type=str, default='/kaggle/input/140k-real-and-fake-faces/valid.csv')
    parser.add_argument('--realfake140k_test_csv', type=str, default='/kaggle/input/140k-real-and-fake-faces/test.csv')
    parser.add_argument('--realfake140k_root_dir', type=str, default='/kaggle/input/140k-real-and-fake-faces')
    
    # 190K Dataset
    parser.add_argument('--realfake190k_root_dir', type=str, default='/kaggle/input/deepfake-and-real-images/Dataset')
    
    # 200K Dataset
    parser.add_argument('--realfake200k_train_csv', type=str, default='/kaggle/input/200k-real-and-fake-faces/train_labels.csv')
    parser.add_argument('--realfake200k_val_csv', type=str, default='/kaggle/input/200k-real-and-fake-faces/val_labels.csv')
    parser.add_argument('--realfake200k_test_csv', type=str, default='/kaggle/input/200k-real-and-fake-faces/test_labels.csv')
    parser.add_argument('--realfake200k_root_dir', type=str, default='/kaggle/input/200k-real-and-fake-faces')
    
    # 330K Dataset
    parser.add_argument('--realfake330k_root_dir', type=str, default='/kaggle/input/deepfake-dataset')
    
    # Data
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
    parser.add_argument('--temperature', '--T', default=4.0, type=float, help='Temperature for KD')
    
    # Fine-tuning
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_lr', type=float, default=0.01)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/pdd_checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()

def train_worker(rank, world_size, args):
    """Training function for each process."""
    
    # Setup DDP
    if args.use_ddp:
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Set seed for reproducibility
    set_seed(args.seed + rank)
    
    is_main_process = (not args.use_ddp) or (rank == 0)
    
    if is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
    
    NUM_CLASSES = 1
    
    if is_main_process:
        print(f"Device: {device}")
        print(f"Using DDP: {args.use_ddp}")
        if args.use_ddp:
            print(f"World Size: {world_size}")
        print(f"Dataset: {args.dataset_mode}")
        print(f"Task: Binary Face Classification")
        print(f"Student Model: ResNet18 (1 output)")
        print(f"Teacher Model: ResNet50 (1 output)")
    
    # Load data based on dataset_mode
    if is_main_process:
        print(f"\nLoading {args.dataset_mode} Dataset...")
    
    dataset_params = {
        'dataset_mode': args.dataset_mode,
        'train_batch_size': args.batch_size,
        'eval_batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'ddp': args.use_ddp
    }
    
    if args.dataset_mode == 'rvf10k':
        dataset_params.update({
            'rvf10k_train_csv': args.rvf10k_train_csv,
            'rvf10k_valid_csv': args.rvf10k_valid_csv,
            'rvf10k_root_dir': args.rvf10k_root_dir
        })
    elif args.dataset_mode == '140k':
        dataset_params.update({
            'realfake140k_train_csv': args.realfake140k_train_csv,
            'realfake140k_valid_csv': args.realfake140k_valid_csv,
            'realfake140k_test_csv': args.realfake140k_test_csv,
            'realfake140k_root_dir': args.realfake140k_root_dir
        })
    elif args.dataset_mode == '190k':
        dataset_params.update({
            'realfake190k_root_dir': args.realfake190k_root_dir
        })
    elif args.dataset_mode == '200k':
        dataset_params.update({
            'realfake200k_train_csv': args.realfake200k_train_csv,
            'realfake200k_val_csv': args.realfake200k_val_csv,
            'realfake200k_test_csv': args.realfake200k_test_csv,
            'realfake200k_root_dir': args.realfake200k_root_dir
        })
    elif args.dataset_mode == '330k':
        dataset_params.update({
            'realfake330k_root_dir': args.realfake330k_root_dir
        })
    
    dataset_selector = Dataset_selector(**dataset_params)
    train_loader = dataset_selector.loader_train
    test_loader = dataset_selector.loader_test
    
    # Create models
    if is_main_process:
        print("\nCreating models...")
    
    student = resnet18(num_classes=NUM_CLASSES).to(device)
    teacher = resnet50(num_classes=NUM_CLASSES).to(device)
    
    if is_main_process:
        print(f"Student (ResNet18) parameters: {sum(p.numel() for p in student.parameters()):,}")
        print(f"Teacher (ResNet50) parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # Wrap models with DDP
    if args.use_ddp:
        student = DDP(student, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        teacher = DDP(teacher, device_ids=[rank], output_device=rank)
    
    # Load teacher
    if is_main_process:
        print("\nLoading teacher model...")
    
    if not os.path.exists(args.teacher_checkpoint):
        if is_main_process:
            print(f"✗ ERROR: Teacher checkpoint not found at {args.teacher_checkpoint}")
        if args.use_ddp:
            cleanup_ddp()
        return
    
    # Load teacher on all processes
    teacher_module = teacher.module if args.use_ddp else teacher
    teacher_module = load_teacher_model(teacher_module, args.teacher_checkpoint, device)
    teacher.eval()
    
    # Evaluate teacher
    if is_main_process:
        print("\nEvaluating teacher model...")
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Teacher Evaluation", leave=False, disable=not is_main_process):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = teacher(inputs).squeeze(1)
            preds = (outputs > 0).long()
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    
    # Aggregate results across all processes if using DDP
    if args.use_ddp:
        correct_tensor = torch.tensor(correct).to(device)
        total_tensor = torch.tensor(total).to(device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    teacher_acc = 100. * correct / total
    if is_main_process:
        print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
    
    # Phase 1: PDD
    if is_main_process:
        print("\n" + "="*70)
        print("PHASE 1: Pruning During Distillation (50 epochs)")
        print("="*70)
    
    start_time = time.time()
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args, rank=rank if args.use_ddp else 0)
    trainer.train()
    phase1_time = time.time() - start_time
    
    if is_main_process:
        student_module = student.module if args.use_ddp else student
        save_path = os.path.join(args.save_dir, f'student_resnet18_{args.dataset_mode}_with_masks.pth')
        save_checkpoint({
            'state_dict': student_module.state_dict(),
            'masks': trainer.get_masks(),
            'args': args
        }, save_path)
        print(f"✓ Saved to {save_path}")
        print(f"⏱ Phase 1 Time: {phase1_time/60:.2f} minutes")
    
    # Wait for all processes to finish phase 1
    if args.use_ddp:
        dist.barrier()
    
    # Phase 2: Prune (only on main process)
    if is_main_process:
        print("\n" + "="*70)
        print("PHASE 2: Pruning Model")
        print("="*70)
        
        start_time = time.time()
        student_module = student.module if args.use_ddp else student
        pruner = ModelPruner(student_module, trainer.get_masks())
        pruned_student = pruner.prune()
        phase2_time = time.time() - start_time
        
        orig_params, pruned_params = pruner.get_params_count()
        orig_flops, pruned_flops = pruner.get_flops_count()
        
        params_red = (1 - pruned_params / orig_params) * 100
        flops_red = (1 - pruned_flops / orig_flops) * 100
        
        print(f"\nCompression Results:")
        print(f"Parameters: {orig_params:,} → {pruned_params:,} ({params_red:.2f}% reduction)")
        print(f"FLOPs: {orig_flops:,} → {pruned_flops:,} ({flops_red:.2f}% reduction)")
        print(f"⏱ Phase 2 Time: {phase2_time:.2f} seconds")
    else:
        phase2_time = 0
        params_red = 0
        flops_red = 0
    
    # Wait for pruning to complete
    if args.use_ddp:
        dist.barrier()
    
    # Phase 3: Fine-tune
    if is_main_process:
        print("\n" + "="*70)
        print("PHASE 3: Fine-tuning Pruned Model (100 epochs)")
        print("="*70)
    
    start_time = time.time()
    
    # Load pruned model on all processes
    if is_main_process:
        pruned_student = pruned_student.to(device)
    else:
        # Other processes need to load the pruned model
        student_module = student.module if args.use_ddp else student
        pruner = ModelPruner(student_module, trainer.get_masks())
        pruned_student = pruner.prune().to(device)
    
    # Wrap with DDP for fine-tuning
    if args.use_ddp:
        pruned_student = DDP(pruned_student, device_ids=[rank], output_device=rank)
    
    pruned_module = pruned_student.module if args.use_ddp else pruned_student
    
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
        pruned_student.train()
        correct = 0
        total = 0
        train_loss = 0.0
        
        # Training with progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.finetune_epochs} [Train]", 
                    leave=False, disable=not is_main_process)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.float()
            
            optimizer.zero_grad()
            outputs = pruned_student(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            preds = (outputs > 0).long()
            total += targets.size(0)
            correct += preds.eq(targets.long()).sum().item()
            train_loss += loss.item()
            
            if is_main_process:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                'acc': f'{100.*correct/total:.2f}%'})
        
        # Aggregate training metrics
        if args.use_ddp:
            correct_tensor = torch.tensor(correct).to(device)
            total_tensor = torch.tensor(total).to(device)
            loss_tensor = torch.tensor(train_loss).to(device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            correct = correct_tensor.item()
            total = total_tensor.item()
            train_loss = loss_tensor.item()
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate
        pruned_student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.finetune_epochs} [Test]", 
                       leave=False, disable=not is_main_process)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pruned_student(inputs).squeeze(1)
                preds = (outputs > 0).long()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
                
                if is_main_process:
                    pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        
        # Aggregate test metrics
        if args.use_ddp:
            correct_tensor = torch.tensor(correct).to(device)
            total_tensor = torch.tensor(total).to(device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        test_acc = 100. * correct / total
        
        if is_main_process:
            print(f"Epoch [{epoch+1}/{args.finetune_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': pruned_module.state_dict(),
                    'accuracy': test_acc,
                    'params_reduction': params_red,
                    'flops_reduction': flops_red,
                    'args': args,
                    'dataset': args.dataset_mode
                }, os.path.join(args.save_dir, f'pruned_resnet18_{args.dataset_mode}_best.pth'))
        
        scheduler.step()
        
        # Synchronize after each epoch
        if args.use_ddp:
            dist.barrier()
    
    phase3_time = time.time() - start_time
    total_time = phase1_time + phase2_time + phase3_time
    
    if is_main_process:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Dataset: {args.dataset_mode}")
        print(f"Teacher (ResNet50) Accuracy: {teacher_acc:.2f}%")
        print(f"Best Test Accuracy (Pruned ResNet18): {best_acc:.2f}%")
        print(f"Parameters Reduction: {params_red:.2f}%")
        print(f"FLOPs Reduction: {flops_red:.2f}%")
        print(f"\n Training Time Breakdown:")
        print(f"  Phase 1 (PDD): {phase1_time/60:.2f} minutes")
        print(f"  Phase 2 (Pruning): {phase2_time:.2f} seconds")
        print(f"  Phase 3 (Fine-tuning): {phase3_time/60:.2f} minutes")
        print(f"  Total Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print("="*70 + "\n")
    
    # Cleanup
    if args.use_ddp:
        cleanup_ddp()

def main():
    args = parse_args()
    
    if args.use_ddp:
        # Launch multiple processes for DDP
        import torch.multiprocessing as mp
        world_size = args.world_size
        print(f"Launching DDP training with {world_size} GPUs...")
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU/CPU training
        train_worker(0, 1, args)

if __name__ == '__main__':
    main()
