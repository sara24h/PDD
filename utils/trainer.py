import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import time
from tqdm import tqdm

class PDDTrainer:
    def __init__(self, student, teacher, train_loader, test_loader, device, args, rank, world_size):
        self.student = student
        self.teacher = teacher.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        # Initialize masks
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
        
        # Access the underlying module (unwrap DDP)
        model = self.student.module if hasattr(self.student, 'module') else self.student
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device) - 1.2,
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
        """Forward pass with masks applied"""
        model = self.student.module if hasattr(self.student, 'module') else self.student
        
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
        if self.is_main:
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
            print(f"Training on {self.world_size} GPUs")
            print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            
            self.student.train()
            self.train_loader.sampler.set_epoch(epoch)
            
            train_loss = torch.tensor(0.0).to(self.device)
            kd_loss_total = torch.tensor(0.0).to(self.device)
            ce_loss_total = torch.tensor(0.0).to(self.device)
            correct = torch.tensor(0.0).to(self.device)
            total = torch.tensor(0.0).to(self.device)
            
            # Progress bar only on main process
            if self.is_main:
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
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
                
                train_loss += total_loss
                kd_loss_total += kd_loss
                ce_loss_total += ce_loss
                
                predicted = (student_logits > 0).float()
                total += targets.size(0)
                correct += predicted.eq(targets).sum()
                
                if self.is_main:
                    pbar.set_postfix({
                        'loss': f'{total_loss.item():.4f}',
                        'kd': f'{kd_loss.item():.4f}',
                        'ce': f'{ce_loss.item():.4f}'
                    })
            
            # Aggregate metrics across GPUs
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(kd_loss_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(ce_loss_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            
            train_acc = 100. * correct.item() / total.item()
            test_acc = self.evaluate()
            self.scheduler.step()
            
            pruning_ratio = self._calculate_pruning_ratio()
            
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            
            if self.is_main:
                print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
                print(f"Train: Loss={train_loss.item()/len(self.train_loader):.4f}, Acc={train_acc:.2f}%")
                print(f"Test: Acc={test_acc:.2f}%")
                print(f"Losses: KD={kd_loss_total.item()/len(self.train_loader):.4f}, "
                      f"CE={ce_loss_total.item()/len(self.train_loader):.4f}")
                print(f"Pruning Ratio: {pruning_ratio:.2f}%")
                print(f"Time: {epoch_time:.1f}s | Elapsed: {elapsed_time/60:.1f}min")
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    self._print_mask_stats()
            
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_masks = {name: mask.clone().detach()
                                  for name, mask in self.masks.items()}
                if self.is_main and ((epoch + 1) % 5 == 0 or epoch == 0):
                    print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        if self.best_masks is not None:
            for name in self.masks.keys():
                self.masks[name].data = self.best_masks[name].data
        
        if self.is_main:
            print("\n" + "="*70)
            print("Training Complete!")
            print(f"Best Accuracy: {self.best_acc:.2f}%")
            print(f"Final Pruning: {self._calculate_pruning_ratio():.2f}%")
            print(f"Total Time: {(time.time() - start_time)/60:.1f} minutes")
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
        correct = torch.tensor(0.0).to(self.device)
        total = torch.tensor(0.0).to(self.device)
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self._forward_with_masks(inputs)
                
                predicted = (outputs > 0).float()
                targets = targets.float().unsqueeze(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum()
        
        # Aggregate across GPUs
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        
        return 100. * correct.item() / total.item()

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
        
        if self.is_main:
            print(f"\n🔍 Generating binary masks based on raw mask values\n")
        
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
            
            if self.is_main:
                print(f"{name:30s}: {int(kept):3d}/{int(total):3d} kept | "
                      f"Score: min={score_min:.3f}, mean={score_mean:.3f}, max={score_max:.3f}")
        
        return binary_masks
