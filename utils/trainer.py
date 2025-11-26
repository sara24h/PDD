import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import time

class PDDTrainer:
    def __init__(self, student, teacher, train_loader, test_loader, device, args, rank, world_size):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.masks = self._initialize_masks()
        
        mask_params = list(self.masks.values())
        
        self.optimizer = torch.optim.SGD(
            [
                {'params': self.student.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': mask_params, 'lr': args.lr * 5, 'weight_decay': 0.0}
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
        
        # دسترسی به مدل اصلی از DDP wrapper
        base_model = self.student.module if hasattr(self.student, 'module') else self.student
        
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device) - 1.1,
                    requires_grad=True
                )
                masks[name] = mask
        
        return masks

    def _approx_sign(self, x):
        """ApproxSign function"""
        return torch.where(x < -1, torch.zeros_like(x),
                          torch.where(x < 0, (x + 1)**2 / 2,
                                    torch.where(x < 1, (2*x - x**2 + 1)/2, 
                                               torch.ones_like(x))))

    def _forward_with_masks(self, x):
        """Forward pass با اعمال ماسک‌ها"""
        base_model = self.student.module if hasattr(self.student, 'module') else self.student
        
        # Conv1
        out = base_model.conv1(x)
        if 'conv1' in self.masks:
            mask = self._approx_sign(self.masks['conv1'])
            out = out * mask
        out = base_model.bn1(out)
        out = F.relu(out)
        
        # Process each stage
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(base_model, layer_name)
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
                
                # Shortcut
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
        out = base_model.fc(out)
        
        return out

    def train(self):
        if self.is_main_process:
            print("\n" + "="*70)
            print("Starting PDD Training with DDP")
            print("="*70)
            print(f"World Size (GPUs): {self.world_size}")
            print(f"Temperature: {self.args.temperature}")
            print(f"Alpha (KD weight): {self.args.alpha}")
            print(f"Learning Rate: {self.args.lr}")
            print(f"Epochs: {self.args.epochs}")
            print(f"Total Masks: {len(self.masks)}")
            print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            # تنظیم epoch برای sampler
            self.train_loader.sampler.set_epoch(epoch)
            
            self.student.train()
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            correct = 0
            total = 0
            
            # نوار پیشرفت فقط برای main process
            if self.is_main_process:
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
                start_time = time.time()
            else:
                pbar = self.train_loader
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.float().unsqueeze(1)
                
                self.optimizer.zero_grad()
                
                # Student outputs with masks
                student_logits = self._forward_with_masks(inputs)
                
                # Teacher outputs
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
                
                # محاسبه accuracy
                predicted = (student_logits > 0).float()
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # به‌روزرسانی نوار پیشرفت
                if self.is_main_process:
                    current_acc = 100. * correct / total
                    pbar.set_postfix({
                        'loss': f'{total_loss.item():.4f}',
                        'acc': f'{current_acc:.2f}%',
                        'kd': f'{kd_loss.item():.4f}',
                        'ce': f'{ce_loss.item():.4f}'
                    })
            
            # محاسبه میانگین‌های epoch
            train_acc = 100. * correct / total
            
            # همگام‌سازی metrics بین GPU ها
            metrics = torch.tensor([train_loss, kd_loss_total, ce_loss_total, correct, total], 
                                  device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            if self.is_main_process:
                train_loss = metrics[0].item() / (len(self.train_loader) * self.world_size)
                kd_loss_avg = metrics[1].item() / (len(self.train_loader) * self.world_size)
                ce_loss_avg = metrics[2].item() / (len(self.train_loader) * self.world_size)
                train_acc = 100. * metrics[3].item() / metrics[4].item()
            
            # ارزیابی
            test_acc = self.evaluate()
            
            self.scheduler.step()
            
            if self.is_main_process:
                pruning_ratio = self._calculate_pruning_ratio()
                elapsed = time.time() - start_time
                
                print(f"\n{'='*70}")
                print(f"Epoch [{epoch+1}/{self.args.epochs}] - Time: {elapsed:.1f}s")
                print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
                print(f"Test: Acc={test_acc:.2f}%")
                print(f"Losses: KD={kd_loss_avg:.4f}, CE={ce_loss_avg:.4f}")
                print(f"Pruning Ratio: {pruning_ratio:.2f}%")
                print(f"{'='*70}")
                
                self._print_mask_stats()
                
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    self.best_masks = {name: mask.clone().detach()
                                     for name, mask in self.masks.items()}
                    print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        # بازگرداندن بهترین ماسک‌ها
        if self.is_main_process and self.best_masks is not None:
            for name in self.masks.keys():
                self.masks[name].data = self.best_masks[name].data
        
        if self.is_main_process:
            print("\n" + "="*70)
            print("Training Complete!")
            print(f"Best Accuracy: {self.best_acc:.2f}%")
            print(f"Final Pruning: {self._calculate_pruning_ratio():.2f}%")
            print("="*70 + "\n")

    def _print_mask_stats(self):
        """نمایش آمار ماسک‌ها"""
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
        """ارزیابی مدل"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self._forward_with_masks(inputs)
                
                predicted = (outputs > 0).float()
                targets = targets.float().unsqueeze(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # همگام‌سازی نتایج بین GPU ها
        metrics = torch.tensor([correct, total], device=self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        accuracy = 100. * metrics[0].item() / metrics[1].item()
        return accuracy

    def _calculate_pruning_ratio(self):
        """محاسبه نسبت هرس"""
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
        """تولید ماسک‌های باینری نهایی"""
        binary_masks = {}
        
        if self.is_main_process:
            print(f"\n🔍 Generating binary masks")
            print()
        
        for name, mask in self.masks.items():
            raw_mask = mask.detach().squeeze()
            score = self._approx_sign(mask).detach().squeeze()
            binary_mask = (score > 0.0).float()
            binary_masks[name] = binary_mask
            
            if self.is_main_process:
                kept = binary_mask.sum().item()
                total = binary_mask.numel()
                score_min = score.min().item()
                score_max = score.max().item()
                score_mean = score.mean().item()
                
                print(f"{name:30s}: {int(kept):3d}/{int(total):3d} kept | "
                      f"Score: min={score_min:.3f}, mean={score_mean:.3f}, max={score_max:.3f}")
        
        return binary_masks
