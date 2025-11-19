import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PDDTrainer:
    """
    Pruning During Distillation (PDD) Trainer
    Balanced strategy for achieving ~30% pruning as per paper
    """
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Initialize masks
        self.masks = self._initialize_masks()
        
        # ✅ Single optimizer for both model and masks
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
        """
        Initialize learnable masks with balanced distribution
        
        Key insight from paper results (~32% pruning):
        - Need masks distributed around 0 to allow both pruning and keeping
        - Larger variance allows some masks to go below -1 (pruned)
        """
        masks = {}
        
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                # ✅ Initialize around 0 with moderate variance
                # mean=0: balanced (50% chance to prune/keep)
                # std=0.5: allows ~15-20% to start below -1 (pruned)
                # This gives natural pruning distribution
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device) * 0.5,
                    requires_grad=True
                )
                masks[name] = mask
        
        return masks

    def _approx_sign(self, x):
        """
        Differentiable approximation of sign function
        
        Equation (2) from the paper:
                    { 0                    if x < -1
        ApproxSign = { (x+1)²/2            if -1 ≤ x < 0
                    { (2x - x² + 1)/2      if 0 ≤ x < 1
                    { 1                    otherwise
        """
        result = torch.zeros_like(x)
        
        # x < -1: output = 0 (PRUNED)
        mask1 = (x < -1).float()
        result = result + mask1 * 0.0
        
        # -1 ≤ x < 0: output = (x+1)²/2
        mask2 = ((x >= -1) & (x < 0)).float()
        result = result + mask2 * ((x + 1) ** 2 / 2)
        
        # 0 ≤ x < 1: output = (2x - x² + 1)/2
        mask3 = ((x >= 0) & (x < 1)).float()
        result = result + mask3 * (2 * x - x**2 + 1) / 2
        
        # x ≥ 1: output = 1 (KEPT)
        mask4 = (x >= 1).float()
        result = result + mask4 * 1.0
        
        return result

    def _forward_with_masks(self, x):
        """
        Forward pass with mask application (Equation 3)
        """
        # Conv1 + BN + ReLU
        out = self.student.conv1(x)
        out = self.student.bn1(out)
        out = F.relu(out)
        
        # Apply mask to conv1 output
        if 'conv1' in self.masks:
            mask = self._approx_sign(self.masks['conv1'])
            out = out * mask
        
        # Process each stage (layer1, layer2, layer3)
        for layer_name in ['layer1', 'layer2', 'layer3']:
            layer = getattr(self.student, layer_name)
            for i, block in enumerate(layer):
                identity = out
                
                # Conv1 of block
                out = block.conv1(out)
                out = block.bn1(out)
                out = F.relu(out)
                
                # Apply mask to conv1 output
                mask_name = f'{layer_name}.{i}.conv1'
                if mask_name in self.masks:
                    mask = self._approx_sign(self.masks[mask_name])
                    out = out * mask
                
                # Conv2 of block
                out = block.conv2(out)
                out = block.bn2(out)
                
                # Apply mask to conv2 output
                mask_name = f'{layer_name}.{i}.conv2'
                if mask_name in self.masks:
                    mask = self._approx_sign(self.masks[mask_name])
                    out = out * mask
                
                # Shortcut connection
                if len(block.shortcut) > 0:
                    identity = block.shortcut(identity)
                
                # Add residual and apply ReLU
                out += identity
                out = F.relu(out)
        
        # Global average pooling
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        # Linear classifier
        out = self.student.linear(out)
        
        return out

    def train(self):
        """
        Train with pruning during distillation
        Equation (4): L_total = L(z̃_s, z_t) + CE(z̃_s, Y)
        """
        print("\n" + "="*70)
        print("Starting Pruning During Distillation (PDD) - Balanced Strategy")
        print("="*70)
        print(f"Temperature:        {self.args.temperature}")
        print(f"Alpha (distill):    {self.args.alpha}")
        print(f"Learning Rate:      {self.args.lr}")
        print(f"Epochs:             {self.args.epochs}")
        print(f"LR Decay:           {self.args.lr_decay_epochs}")
        print(f"Total Masks:        {len(self.masks)}")
        print("="*70 + "\n")
        
        # Print initial mask distribution
        self._print_detailed_mask_stats("Initial")
        
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
                
                # Student outputs with masks (Equation 3)
                student_outputs = self._forward_with_masks(inputs)
                
                # Teacher outputs (no gradients needed)
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs)
                
                # ✅ Calculate losses - ONLY KD + CE (Equation 4)
                
                # 1. Classification loss
                ce_loss = self.criterion(student_outputs, targets)
                
                # 2. Knowledge distillation loss
                soft_teacher = F.softmax(teacher_outputs / self.args.temperature, dim=1)
                soft_student = F.log_softmax(student_outputs / self.args.temperature, dim=1)
                kd_loss = self.kd_criterion(soft_student, soft_teacher) * (self.args.temperature ** 2)
                
                # ✅ Total loss (Equation 4) - NO REGULARIZATION
                total_loss = self.args.alpha * kd_loss + (1 - self.args.alpha) * ce_loss
                
                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += total_loss.item()
                kd_loss_total += kd_loss.item()
                ce_loss_total += ce_loss.item()
                
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            
            # Evaluation phase
            test_acc = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()

            # Calculate pruning ratio
            pruning_ratio = self._calculate_pruning_ratio()
            
            # Print epoch statistics
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train: Loss={train_loss/len(self.train_loader):.4f}, Acc={train_acc:.2f}%")
            print(f"Test:  Acc={test_acc:.2f}%")
            print(f"Losses: KD={kd_loss_total/len(self.train_loader):.4f}, "
                  f"CE={ce_loss_total/len(self.train_loader):.4f}")
            print(f"Pruning Ratio: {pruning_ratio:.2f}%")
            
            # Print detailed mask statistics
            self._print_detailed_mask_stats(f"Epoch {epoch+1}")
            
            # Save best model based on test accuracy
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_masks = {name: mask.clone().detach() 
                                  for name, mask in self.masks.items()}
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        # Restore best masks after training
        if self.best_masks is not None:
            for name in self.masks.keys():
                self.masks[name].data = self.best_masks[name].data
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        print(f"Final Pruning: {self._calculate_pruning_ratio():.2f}%")
        print("="*70 + "\n")

    def _print_detailed_mask_stats(self, label):
        """Print detailed mask statistics for debugging"""
        raw_values = []
        approx_values = []
        below_minus_one = 0
        total_channels = 0
        
        for mask in self.masks.values():
            raw = mask.detach()
            approx = self._approx_sign(mask).detach()
            
            raw_values.extend(raw.cpu().flatten().tolist())
            approx_values.extend(approx.cpu().flatten().tolist())
            
            below_minus_one += (raw < -1).sum().item()
            total_channels += raw.numel()
        
        raw_values = np.array(raw_values)
        approx_values = np.array(approx_values)
        
        print(f"\n  [{label}] Mask Distribution:")
        print(f"    Raw: mean={raw_values.mean():.3f}, std={raw_values.std():.3f}, "
              f"min={raw_values.min():.3f}, max={raw_values.max():.3f}")
        print(f"    After ApproxSign: mean={approx_values.mean():.3f}, "
              f"min={approx_values.min():.3f}, max={approx_values.max():.3f}")
        print(f"    Channels with raw < -1: {below_minus_one}/{total_channels} "
              f"({100*below_minus_one/total_channels:.1f}%)")

    def evaluate(self):
        """Evaluate the student model on test set"""
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
        Calculate pruning ratio
        Channels with raw mask < -1 are pruned (ApproxSign = 0)
        """
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            raw_mask = mask.detach()
            total_channels += raw_mask.numel()
            # Count channels where raw value < -1 (will be pruned)
            pruned_channels += (raw_mask < -1).sum().item()
        
        if total_channels == 0:
            return 0.0
        
        return 100. * pruned_channels / total_channels

    def get_masks(self):
        """
        Get binary masks for pruning
        Keep channels where raw mask >= -1
        """
        binary_masks = {}
        for name, mask in self.masks.items():
            raw_mask = mask.detach()
            # Keep if raw >= -1, prune if raw < -1
            binary_masks[name] = (raw_mask >= -1).float()
        return binary_masks
