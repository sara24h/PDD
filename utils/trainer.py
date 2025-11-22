import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PDDTrainer:
   
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Initialize masks
        self.masks = self._initialize_masks()
        
        # Single optimizer for both model and masks
        mask_params = list(self.masks.values())
        
        self.optimizer = torch.optim.SGD(
            list(self.student.parameters()) + mask_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean')
        
        # Training stats
        self.best_acc = 0.0
        self.best_masks = None

    def _initialize_masks(self):
        masks = {}
        
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device),
                    requires_grad=True
                )
                masks[name] = mask
        
        return masks

    def _approx_sign(self, x):
        return torch.where(x < -1, torch.zeros_like(x),
                           torch.where(x < 0, (x + 1)**2 / 2,
                                       torch.where(x < 1, (2*x - x**2 + 1)/2, torch.ones_like(x))))

    def _forward_with_masks(self, x):
        # Conv1
        out = self.student.conv1(x)
        if 'conv1' in self.masks:
            mask = self._approx_sign(self.masks['conv1'])
            out = out * mask
        out = self.student.bn1(out)
        out = F.relu(out)
        
        # Process each stage
        for layer_name in ['layer1', 'layer2', 'layer3']:
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
                if len(block.shortcut) > 0:
                    identity = block.shortcut(identity)
                
                out += identity
                out = F.relu(out)
        
        # Global average pooling
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.student.linear(out)
        
        return out

    def train(self):
        print("\n" + "="*70)
        print("Starting Pruning During Distillation (PDD) - EXACT Paper Implementation")
        print("="*70)
        print(f"Temperature: {self.args.temperature}")
        print(f"Alpha (distill): {self.args.alpha}")
        print(f"Learning Rate: {self.args.lr}")
        print(f"Epochs: {self.args.epochs}")
        print(f"LR Decay: {self.args.lr_decay_epochs}")
        print(f"Total Masks: {len(self.masks)}")
        print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            # Training phase
            self.student.train()
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Student outputs with masks
                student_outputs = self._forward_with_masks(inputs)
                
                # Teacher outputs
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs)
                
                # Classification loss
                ce_loss = self.criterion(student_outputs, targets)
                
                # Knowledge distillation loss
                soft_teacher = F.softmax(teacher_outputs / self.args.temperature, dim=1)
                soft_student = F.log_softmax(student_outputs / self.args.temperature, dim=1)
                kd_loss = self.kd_criterion(soft_student, soft_teacher) * (self.args.temperature ** 2)
                
                # Total loss
                total_loss = self.args.alpha * kd_loss + (1 - self.args.alpha) * ce_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
                kd_loss_total += kd_loss.item()
                ce_loss_total += ce_loss.item()
                
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
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
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self._forward_with_masks(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total

    def _calculate_pruning_ratio(self):
        """
        ✅ EXACT as per paper:
        Channels are pruned when raw_mask < -1 (where score = 0)
        """
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
        """
        ✅ FIXED: Generate binary masks based on RAW mask values, not scores
        
        Paper (Page 3461): "a score of 0 indicates that the channel is redundant"
        From ApproxSign: score = 0 when raw_mask < -1
        
        Therefore:
        - Keep channel if raw_mask >= -1 (score > 0)
        - Prune channel if raw_mask < -1 (score = 0)
        """
        binary_masks = {}
        
        print(f"\n🔍 Generating binary masks based on raw mask values")
        print(f"Threshold: raw_mask >= -1.0 (keep), raw_mask < -1.0 (prune)")
        print()
        
        for name, mask in self.masks.items():
            # Get RAW mask values (not ApproxSign scores)
            raw_mask = mask.detach().squeeze()  # [C]
            
            # Apply ApproxSign to get scores for display
            score = self._approx_sign(mask).detach().squeeze()
            
            # ✅ Binary mask based on RAW values (as per paper)
            # Keep if raw_mask >= -1 (where score > 0)
            # Prune if raw_mask < -1 (where score = 0)
            binary_mask = (raw_mask >= -1.0).float()
            
            binary_masks[name] = binary_mask
            
            # Statistics
            kept = binary_mask.sum().item()
            total = binary_mask.numel()
            score_min = score.min().item()
            score_max = score.max().item()
            score_mean = score.mean().item()
            
            print(f"{name:30s}: {int(kept):3d}/{int(total):3d} kept | "
                  f"Score: min={score_min:.3f}, mean={score_mean:.3f}, max={score_max:.3f}")
        
        return binary_masks

    def prune_model(self):
        """
        Generate pruning summary based on learned masks
        """
        binary_masks = self.get_masks()
        
        print("\n" + "="*70)
        print("Model Pruning Summary")
        print("="*70)
        
        total_original = 0
        total_kept = 0
        
        for name, mask in binary_masks.items():
            kept = mask.sum().item()
            total = mask.numel()
            total_original += total
            total_kept += kept
            pruned_ratio = (1 - kept/total) * 100
            print(f"{name:30s}: {int(kept):4d}/{int(total):4d} channels | "
                  f"Pruned: {pruned_ratio:.1f}%")
        
        print("="*70)
        overall_pruned = (1 - total_kept/total_original) * 100
        print(f"Overall: {int(total_kept):4d}/{int(total_original):4d} channels kept | "
              f"Pruned: {overall_pruned:.2f}%")
        print("="*70 + "\n")
        
        return binary_masks
