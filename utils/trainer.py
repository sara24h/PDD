# utils/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class PDDTrainer:
  
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args

        self.masks = self._initialize_masks()

        mask_params = list(self.masks.values())
        
        self.optimizer = torch.optim.SGD(
            [
                {'params': self.student.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': mask_params, 'lr': args.lr * 5.0, 'weight_decay': 0.0}
            ],
            momentum=args.momentum
        )
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.best_acc = 0.0
        self.best_masks = None

    def _initialize_masks(self):
        masks = {}
        
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device) -1.2,
                    requires_grad=True
                )
                masks[name] = mask
        
        return masks

    def _approx_sign(self, x):
        """ApproxSign function from paper"""
        return torch.where(x < -1, torch.zeros_like(x),
                           torch.where(x < 0, (x + 1)**2 / 2,
                                       torch.where(x < 1, (2*x - x**2 + 1)/2, 
                                                   torch.ones_like(x))))

    def _forward_with_masks(self, x):
        # Conv1
        out = self.student.conv1(x)
        if 'conv1' in self.masks:
            mask = self._approx_sign(self.masks['conv1'])
            out = out * mask
        out = self.student.bn1(out)
        out = F.relu(out)
        
        # Process each stage
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.student, layer_name)
            for i, block in enumerate(layer):
                identity = out
                
                # Conv1 of block
                out = block.conv1(out)
                mask_name = f'{layer_name}.{i}.conv1'
                if mask_name in self.masks:
                    mask = self._approx_sign(self.masks[mask_name])
                    out = out * mask
                out = block.bn1(out)
                out = F.relu(out)
                
                # Conv2 of block
                out = block.conv2(out)
                mask_name = f'{layer_name}.{i}.conv2'
                if mask_name in self.masks:
                    mask = self._approx_sign(self.masks[mask_name])
                    out = out * mask
                out = block.bn2(out)
                
                # Shortcut connection
                if block.downsample is not None:
                    identity = block.downsample(identity)
                    shortcut_mask_name = f'{layer_name}.{i}.downsample.0'
                    if shortcut_mask_name in self.masks:
                        shortcut_mask = self._approx_sign(self.masks[shortcut_mask_name])
                        identity = identity * shortcut_mask
                
                out += identity
                out = F.relu(out)
        
        # Global average pooling
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.student.fc(out)
        
        return out

    def train(self):
        print("\n" + "="*70)
        print("Starting PDD Training - Binary Classification with BCE")
        print("="*70)
        print(f"Temperature: {self.args.temperature}")
        print(f"Alpha (KD weight): {self.args.alpha}")
        print(f"Learning Rate: {self.args.lr}")
        print(f"Mask Learning Rate: {self.args.lr * 5.0} (5x higher)")
        print(f"Epochs: {self.args.epochs}")
        print(f"LR Decay: {self.args.lr_decay_epochs}")
        print(f"Total Masks: {len(self.masks)}")
        print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            self.student.train()
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            correct = 0
            total = 0
            
            # Progress bar for training
            pbar = tqdm(self.train_loader, 
                       desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]",
                       leave=False)
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.float().unsqueeze(1)
                
                self.optimizer.zero_grad()
                
                # Student outputs with masks (logits)
                student_logits = self._forward_with_masks(inputs)
                
                # Teacher outputs (logits)
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)

                # Classification Loss
                ce_loss = self.criterion(student_logits, targets)
                
                # Knowledge Distillation Loss
                teacher_probs = torch.sigmoid(teacher_logits / self.args.temperature)
                student_probs = torch.sigmoid(student_logits / self.args.temperature)
                
                kd_loss = F.binary_cross_entropy(student_probs, teacher_probs, reduction='mean') * (self.args.temperature ** 2)

                total_loss = self.args.alpha * kd_loss + (1 - self.args.alpha) * ce_loss
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(list(self.masks.values()), max_norm=5.0)
                
                self.optimizer.step()
                
                train_loss += total_loss.item()
                kd_loss_total += kd_loss.item()
                ce_loss_total += ce_loss.item()
                
                predicted = (student_logits > 0).float()
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'kd': f'{kd_loss.item():.4f}',
                    'ce': f'{ce_loss.item():.4f}'
                })
            
            train_acc = 100. * correct / total
            test_acc = self.evaluate()
            self.scheduler.step()
            
            pruning_ratio = self._calculate_pruning_ratio()
            
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train: Loss={train_loss/len(self.train_loader):.4f}, Acc={train_acc:.2f}%")
            print(f"Test: Acc={test_acc:.2f}%")
            print(f"Losses: KD={kd_loss_total/len(self.train_loader):.4f}, "
                  f"CE={ce_loss_total/len(self.train_loader):.4f}")
            print(f"Pruning Ratio: {pruning_ratio:.2f}%")
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self._print_mask_stats()
            
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_masks = {name: mask.clone().detach()
                                  for name, mask in self.masks.items()}
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        if self.best_masks is not None:
            for name in self.masks.keys():
                self.masks[name].data = self.best_masks[name].data
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        print(f"Final Pruning: {self._calculate_pruning_ratio():.2f}%")
        print("="*70 + "\n")

    def _print_mask_stats(self):
        """Print detailed mask statistics"""
        raw_means = []
        raw_mins = []
        raw_maxs = []
        approx_means = []
        pruned_counts = []
        
        for mask in self.masks.values():
            raw = mask.detach()
            approx = self._approx_sign(mask).detach()
            
            raw_means.append(raw.mean().item())
            raw_mins.append(raw.min().item())
            raw_maxs.append(raw.max().item())
            approx_means.append(approx.mean().item())
            pruned_counts.append((raw < -1).sum().item())
        
        print(f" Raw Masks: Avg={np.mean(raw_means):.3f}, "
              f"Min={np.min(raw_mins):.3f}, Max={np.max(raw_maxs):.3f}")
        print(f" After ApproxSign: Avg={np.mean(approx_means):.3f}")
        print(f" Channels with score=0 (raw<-1): {np.sum(pruned_counts)}")

    def evaluate(self):
        """Evaluate model with masks"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self._forward_with_masks(inputs)
                
                predicted = (outputs > 0).float()
                targets = targets.float().unsqueeze(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        
        return 100. * correct / total

    def _calculate_pruning_ratio(self):
        """Calculate current pruning ratio"""
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            raw_mask = mask.detach()
            total_channels += raw_mask.numel()
            pruned_channels += (raw_mask < -1).sum().item()
        
        if total_channels == 0:
            return 0.0
        
        return 100. * pruned_channels / total_channels

    def get_masks(self):
        """Generate final binary masks for pruning"""
        binary_masks = {}
        
        print(f"\n🔍 Generating binary masks based on raw mask values")
        print()
        
        for name, mask in self.masks.items():
            raw_mask = mask.detach().squeeze()
            score = self._approx_sign(mask).detach().squeeze()
            binary_mask = (score > 0.0).float()
            binary_masks[name] = binary_mask

            kept = binary_mask.sum().item()
            total = binary_mask.numel()
            score_min = score.min().item()
            score_max = score.max().item()
            score_mean = score.mean().item()
            
            print(f"{name:30s}: {int(kept):3d}/{int(total):3d} kept | "
                  f"Score: min={score_min:.3f}, mean={score_mean:.3f}, max={score_max:.3f}")
        
        return binary_masks


# ===================================================================
# تابع اصلی اجرا برای هر پردازش (در حالت DDP)
# ===================================================================
def main_worker(rank, args):
    # Setup DDP
    if args.world_size > 1:
        setup(rank, args.world_size)
    
    set_seed(args.seed + rank)
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 1
    
    if rank == 0:
        print("\n" + "="*70)
        print("PDD: Pruning During Distillation")
        print("="*70)
        print(f"Device: {device}")
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Batch Size: {args.batch_size}")
        print("="*70)
    
    # Load data
    train_loader, test_loader = load_dataset(args, rank)
    
    # Create models
    student = resnet18(num_classes=NUM_CLASSES).to(device)
    teacher = resnet50(num_classes=NUM_CLASSES).to(device)
    
    # Wrap models with DDP
    if args.world_size > 1:
        student = DDP(student, device_ids=[rank])
        teacher = DDP(teacher, device_ids=[rank])

    if rank == 0:
        print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
        print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # Load teacher
    if rank == 0:
        print("\nLoading teacher model...")
        load_teacher_model(teacher.module if args.world_size > 1 else teacher, args.teacher_checkpoint, device)
    
    # Evaluate teacher (only on rank 0)
    if rank == 0:
        print("\nEvaluating teacher model...")
        correct = 0
        total = 0
        teacher_model = teacher.module if args.world_size > 1 else teacher
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating teacher", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = teacher_model(inputs).squeeze(1)
                preds = (outputs > 0).long()
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
        
        teacher_acc = 100. * correct / total
        print(f"Teacher Test Accuracy: {teacher_acc:.2f}%")
    else:
        teacher_acc = 0.0 # مقداردهی اولیه برای رنک‌های دیگر

    # Phase 1: PDD (Pruning During Distillation)
    distillation_start_time = time.time()
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args, rank=rank)
    trainer.train()
    distillation_time = time.time() - distillation_start_time

    # Save student with masks (only on rank 0)
    if rank == 0:
        checkpoint_name = f'student_resnet18_{args.dataset}_with_masks.pth'
        save_path = os.path.join(args.save_dir, checkpoint_name)
        save_checkpoint({
            'state_dict': student.module.state_dict() if args.world_size > 1 else student.state_dict(),
            'masks': trainer.get_masks(),
            'best_acc': trainer.best_acc,
            'args': vars(args),
        }, save_path)
        print(f"✓ Saved checkpoint to {save_path}")
    
    # Phase 2: Prune (only on rank 0)
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 2: Pruning Model")
        print("="*70)
        
        pruner = ModelPruner(student.module if args.world_size > 1 else student, trainer.get_masks())
        pruned_student = pruner.prune()
        
        orig_params, pruned_params = pruner.get_params_count()
        orig_flops, pruned_flops = pruner.get_flops_count()
        params_red = (1 - pruned_params / orig_params) * 100
        flops_red = (1 - pruned_flops / orig_flops) * 100
        
        print(f"\nParameters: {orig_params:,} → {pruned_params:,} ({params_red:.2f}% reduction)")
        print(f"FLOPs: {orig_flops:,} → {pruned_flops:,} ({flops_red:.2f}% reduction)\n")
        
        # ذخیره مدل پرین‌شده برای بارگذاری در تمام رنک‌ها
        temp_pruned_path = os.path.join(args.save_dir, 'temp_pruned.pth')
        torch.save(pruned_student.state_dict(), temp_pruned_path)

    if args.world_size > 1:
        dist.barrier()

    # Load pruned model on all processes
    if rank != 0:
        temp_pruned_path = os.path.join(args.save_dir, 'temp_pruned.pth')
        pruned_student = resnet18(num_classes=NUM_CLASSES).to(device)
        pruned_student.load_state_dict(torch.load(temp_pruned_path, map_location=device))
    else:
        # رنک 0 هم مدل را از فایل بارگذاری می‌کند تا یکپارچگی حفظ شود
        temp_pruned_path = os.path.join(args.save_dir, 'temp_pruned.pth')
        pruned_student = resnet18(num_classes=NUM_CLASSES).to(device)
        pruned_student.load_state_dict(torch.load(temp_pruned_path, map_location=device))

    if args.world_size > 1:
        pruned_student = DDP(pruned_student, device_ids=[rank])
    
    # Phase 3: Fine-tune
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 3: Fine-tuning Pruned Model")
        print("="*70)
    
    optimizer = torch.optim.SGD(pruned_student.parameters(), lr=args.finetune_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    
    finetune_start_time = time.time()
    
    for epoch in range(args.finetune_epochs):
        pruned_student.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{args.finetune_epochs}", leave=False, disable=rank != 0)
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device).float()
            optimizer.zero_grad()
            outputs = pruned_student(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs > 0).long()
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        
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
        
        if rank == 0 and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch [{epoch+1:3d}/{args.finetune_epochs}] Train Acc: {100.*correct/total:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if rank == 0 and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': pruned_student.module.state_dict() if args.world_size > 1 else pruned_student.state_dict(),
                'accuracy': test_acc,
                'args': vars(args),
            }, os.path.join(args.save_dir, f'pruned_resnet18_{args.dataset}_best.pth'))
        
        scheduler.step()
    
    finetune_time = time.time() - finetune_start_time

    # Final Results (only on rank 0)
    if rank == 0:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Teacher Accuracy: {teacher_acc:.2f}%")
        print(f"Student Best Accuracy after Distillation: {trainer.best_acc:.2f}%")
        print(f"Pruned Student Best Accuracy after Fine-tuning: {best_acc:.2f}%")
        print(f"Parameters Reduction: {params_red:.2f}%")
        print(f"FLOPs Reduction: {flops_red:.2f}%")
        print(f"Distillation Time: {distillation_time/60:.2f} minutes")
        print(f"Fine-tuning Time: {finetune_time/60:.2f} minutes")
        print(f"Total Training Time: {(distillation_time + finetune_time)/60:.2f} minutes")
        print("="*70 + "\n")
        
        summary = {
            'dataset': args.dataset,
            'teacher_acc': teacher_acc,
            'student_after_distillation': trainer.best_acc,
            'pruned_student_after_finetune': best_acc,
            'params_reduction': params_red,
            'flops_reduction': flops_red,
            'distillation_time_minutes': distillation_time/60,
            'finetune_time_minutes': finetune_time/60,
            'total_time_minutes': (distillation_time + finetune_time)/60
        }
        
        summary_path = os.path.join(args.save_dir, f'summary_{args.dataset}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✓ Summary saved to {summary_path}\n")
    
    if args.world_size > 1:
        cleanup()


def main():
    args = parse_args()
    
    # ایجاد یک فایل چک‌پوینت ساختگی برای معلم
    if not os.path.exists(args.teacher_checkpoint):
        print(f"Warning: Teacher checkpoint not found at {args.teacher_checkpoint}. Creating a dummy one.")
        os.makedirs(os.path.dirname(args.teacher_checkpoint), exist_ok=True)
        dummy_model = resnet50()
        torch.save(dummy_model.state_dict(), args.teacher_checkpoint)

    if args.world_size > 1:
        mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
    else:
        main_worker(0, args)


if __name__ == '__main__':
    main()
