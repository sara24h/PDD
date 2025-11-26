# utils/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
import os
import time
import json


class PDDTrainer:
    """
    Trainer class for Pruning During Distillation (PDD) with DDP support
    """
    
    def __init__(self, student, teacher, train_loader, test_loader, device, args, rank=0):
        self.rank = rank
        self.device = device
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Move models to device
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        
        # Initialize masks before wrapping with DDP
        self.masks = self._initialize_masks()
        
        # Wrap models with DDP if using distributed training
        if args.world_size > 1:
            self.student = DDP(self.student, device_ids=[rank], output_device=rank)
            self.teacher = DDP(self.teacher, device_ids=[rank], output_device=rank)
        
        # Setup optimizer
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
        """Initialize learnable masks for each convolutional layer"""
        masks = {}
        
        # Get the actual model (unwrap DDP if needed)
        model = self.student.module if isinstance(self.student, DDP) else self.student
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device) - 1.2,
                    requires_grad=True
                )
                masks[name] = mask
        
        return masks

    def _approx_sign(self, x):
        """ApproxSign function from PDD paper"""
        return torch.where(x < -1, torch.zeros_like(x),
                           torch.where(x < 0, (x + 1)**2 / 2,
                                       torch.where(x < 1, (2*x - x**2 + 1)/2, 
                                                   torch.ones_like(x))))

    def _forward_with_masks(self, x):
        """Forward pass with mask application"""
        # Get the actual model (unwrap DDP if needed)
        model = self.student.module if isinstance(self.student, DDP) else self.student
        
        # Conv1
        out = model.conv1(x)
        if 'conv1' in self.masks:
            mask = self._approx_sign(self.masks['conv1'])
            out = out * mask
        out = model.bn1(out)
        out = F.relu(out)
        
        # Process each stage
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
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
        out = model.fc(out)
        
        return out

    def train(self):
        """Main training loop for PDD"""
        if self.rank == 0:
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
            print(f"World Size: {self.args.world_size}")
            print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            # Set sampler epoch for proper shuffling in DDP
            if self.args.world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            self.student.train()
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            correct = 0
            total = 0
            
            # Progress bar (only on rank 0)
            pbar = tqdm(self.train_loader, 
                       desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]",
                       leave=False,
                       disable=self.rank != 0)
            
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
                
                # Update progress bar (only on rank 0)
                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': f'{total_loss.item():.4f}',
                        'acc': f'{100.*correct/total:.2f}%',
                        'kd': f'{kd_loss.item():.4f}',
                        'ce': f'{ce_loss.item():.4f}'
                    })
            
            # Synchronize metrics across all processes
            if self.args.world_size > 1:
                metrics = torch.tensor([train_loss, kd_loss_total, ce_loss_total, correct, total], 
                                       dtype=torch.float32, device=self.device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                train_loss, kd_loss_total, ce_loss_total, correct, total = metrics.tolist()
            
            train_acc = 100. * correct / total
            test_acc = self.evaluate()
            self.scheduler.step()
            
            pruning_ratio = self._calculate_pruning_ratio()
            
            if self.rank == 0:
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
                if self.rank == 0 and ((epoch + 1) % 5 == 0 or epoch == 0):
                    print(f"✓ New best accuracy: {test_acc:.2f}%")
            
            # Synchronize between epochs
            if self.args.world_size > 1:
                dist.barrier()
        
        # Restore best masks
        if self.best_masks is not None:
            for name in self.masks.keys():
                self.masks[name].data = self.best_masks[name].data
        
        if self.rank == 0:
            print("\n" + "="*70)
            print("Training Complete!")
            print(f"Best Accuracy: {self.best_acc:.2f}%")
            print(f"Final Pruning: {self._calculate_pruning_ratio():.2f}%")
            print("="*70 + "\n")

    def _print_mask_stats(self):
        """Print detailed mask statistics (only on rank 0)"""
        if self.rank != 0:
            return
            
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
            pbar = tqdm(self.test_loader, desc="Evaluating", leave=False, disable=self.rank != 0)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self._forward_with_masks(inputs)
                
                predicted = (outputs > 0).float()
                targets = targets.float().unsqueeze(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if self.rank == 0:
                    pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        
        # Synchronize metrics across all processes
        if self.args.world_size > 1:
            metrics = torch.tensor([correct, total], dtype=torch.float32, device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            correct, total = metrics.tolist()
        
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
        
        if self.rank == 0:
            print(f"\n🔍 Generating binary masks based on raw mask values")
            print()
        
        for name, mask in self.masks.items():
            raw_mask = mask.detach().squeeze()
            score = self._approx_sign(mask).detach().squeeze()
            binary_mask = (score > 0.0).float()
            binary_masks[name] = binary_mask

            if self.rank == 0:
                kept = binary_mask.sum().item()
                total = binary_mask.numel()
                score_min = score.min().item()
                score_max = score.max().item()
                score_mean = score.mean().item()
                
                print(f"{name:30s}: {int(kept):3d}/{int(total):3d} kept | "
                      f"Score: min={score_min:.3f}, mean={score_mean:.3f}, max={score_max:.3f}")
        
        return binary_masks


def setup(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def save_checkpoint(state, filepath):
    """Save checkpoint to file"""
    torch.save(state, filepath)


def load_teacher_model(teacher, checkpoint_path, device):
    """Load teacher model from checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            teacher.load_state_dict(checkpoint['state_dict'])
        else:
            teacher.load_state_dict(checkpoint)
        print(f"✓ Teacher loaded from {checkpoint_path}")
    else:
        print(f"Warning: Teacher checkpoint not found at {checkpoint_path}")


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
