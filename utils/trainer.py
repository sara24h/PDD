import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnableMask(nn.Module):
    """Learnable mask module with improved forward computation"""
    def __init__(self, num_channels, device):
        super(LearnableMask, self).__init__()
        # مقداردهی اولیه با توزیع نرمال
        self.mask = nn.Parameter(
            torch.randn(1, num_channels, 1, 1, device=device),
            requires_grad=True
        )
    
    def forward(self):
        """
        Compute mask value between 0 and 1 from raw mask values.
        Raw mask values are typically in range [-inf, +inf]
        We map them to [-1, 1] using tanh, then shift and scale to [0, 1]
        """
        # Step 1: Map raw values to [-1, 1] using tanh
        normalized = torch.tanh(self.mask)
        
        # Step 2: Shift and scale to [0, 1]
        # (normalized + 1) / 2 maps [-1, 1] to [0, 1]
        mask_01 = (normalized + 1.0) / 2.0
        
        return mask_01


class PDDTrainer:
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Initialize learnable masks
        self.mask_modules = self._initialize_masks()
        
        # Optimizer - بهینه‌سازی پارامترهای مدل و ماسک‌ها
        mask_params = []
        for mask_module in self.mask_modules.values():
            mask_params.extend(mask_module.parameters())
        
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
        """Initialize learnable mask modules for each convolutional layer"""
        mask_modules = nn.ModuleDict()
        
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                # ایجاد یک mask module برای این لایه
                mask_modules[name.replace('.', '_')] = LearnableMask(
                    module.out_channels, 
                    self.device
                )
        
        return mask_modules

    def _get_mask_for_layer(self, layer_name):
        """Get the computed mask (0-1 values) for a specific layer"""
        mask_key = layer_name.replace('.', '_')
        if mask_key in self.mask_modules:
            return self.mask_modules[mask_key]()
        return None

    def _forward_with_masks(self, x):
        """Forward pass with mask application"""
        # Conv1
        out = self.student.conv1(x)
        out = self.student.bn1(out)
        out = F.relu(out)
        
        # Apply mask to conv1
        mask = self._get_mask_for_layer('conv1')
        if mask is not None:
            out = out * mask
        
        # Process layers
        for layer_name in ['layer1', 'layer2', 'layer3']:
            layer = getattr(self.student, layer_name)
            for i, block in enumerate(layer):
                identity = out
                
                # Conv1 of block
                out = block.conv1(out)
                out = block.bn1(out)
                out = F.relu(out)
                
                # Apply mask
                mask = self._get_mask_for_layer(f'{layer_name}.{i}.conv1')
                if mask is not None:
                    out = out * mask
                
                # Conv2 of block
                out = block.conv2(out)
                out = block.bn2(out)
                
                # Apply mask
                mask = self._get_mask_for_layer(f'{layer_name}.{i}.conv2')
                if mask is not None:
                    out = out * mask
                
                # Shortcut
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
        """Train the student model with pruning during distillation"""
        print("\n" + "="*70)
        print("Starting Pruning During Distillation (PDD)")
        print("="*70)
        print(f"Temperature:        {self.args.temperature}")
        print(f"Alpha (distill):    {self.args.alpha}")
        print(f"Learning Rate:      {self.args.lr}")
        print(f"Epochs:             {self.args.epochs}")
        print(f"LR Decay:           {self.args.lr_decay_epochs}")
        print(f"Total Masks:        {len(self.mask_modules)}")
        print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            # Train
            self.student.train()
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            reg_loss_total = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Student outputs with masks
                student_outputs = self._forward_with_masks(inputs)
                
                # Teacher outputs (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs)
                
                # Calculate losses
                # 1. Classification loss (Cross Entropy)
                ce_loss = self.criterion(student_outputs, targets)
                
                # 2. Knowledge distillation loss (KL Divergence)
                soft_teacher = F.softmax(teacher_outputs / self.args.temperature, dim=1)
                soft_student = F.log_softmax(student_outputs / self.args.temperature, dim=1)
                kd_loss = self.kd_criterion(soft_student, soft_teacher) * (self.args.temperature ** 2)
                
                # 3. Sparsity regularization (L1 on mask values)
                reg_loss = 0.0
                num_masks = 0
                for mask_module in self.mask_modules.values():
                    mask_values = mask_module()
                    reg_loss += torch.sum(mask_values)
                    num_masks += mask_values.numel()
                
                if num_masks > 0:
                    reg_loss = reg_loss / num_masks
                
                # Total loss
                total_loss = (self.args.alpha * kd_loss + 
                             (1 - self.args.alpha) * ce_loss + 
                             0.1 * reg_loss)
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += total_loss.item()
                kd_loss_total += kd_loss.item()
                ce_loss_total += ce_loss.item()
                reg_loss_total += reg_loss.item()
                
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            
            # Evaluate
            test_acc = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()

            # Calculate pruning statistics
            pruning_ratio = self._calculate_pruning_ratio()
            
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train: Loss={train_loss/len(self.train_loader):.4f}, Acc={train_acc:.2f}%")
            print(f"Test:  Acc={test_acc:.2f}%")
            print(f"Losses: KD={kd_loss_total/len(self.train_loader):.4f}, "
                  f"CE={ce_loss_total/len(self.train_loader):.4f}, "
                  f"Reg={reg_loss_total/len(self.train_loader):.6f}")
            print(f"Pruning Ratio: {pruning_ratio:.2f}%")
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_masks = {}
                for name, mask_module in self.mask_modules.items():
                    self.best_masks[name] = mask_module.mask.clone().detach()
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        # Restore best masks
        if self.best_masks is not None:
            for name, mask_module in self.mask_modules.items():
                if name in self.best_masks:
                    mask_module.mask.data = self.best_masks[name].data
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        print(f"Final Pruning: {self._calculate_pruning_ratio():.2f}%")
        print("="*70 + "\n")

    def evaluate(self):
        """Evaluate the student model"""
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

    def _calculate_pruning_ratio(self, threshold=0.5):
        """Calculate the current pruning ratio based on threshold"""
        total_channels = 0
        pruned_channels = 0
        
        for mask_module in self.mask_modules.values():
            mask_values = mask_module()
            total_channels += mask_values.numel()
            # Channels with mask value < threshold are considered pruned
            pruned_channels += (mask_values < threshold).sum().item()
        
        if total_channels == 0:
            return 0.0
        
        return 100. * pruned_channels / total_channels

    def get_masks(self, threshold=0.5):
        """Get the current masks as binary (0 or 1) based on threshold"""
        binary_masks = {}
        
        for name, mask_module in self.mask_modules.items():
            # Get mask values (0-1 range)
            mask_values = mask_module()
            # Convert to binary based on threshold
            binary_mask = (mask_values > threshold).float()
            # Convert name back to original format
            original_name = name.replace('_', '.')
            binary_masks[original_name] = binary_mask
        
        return binary_masks

    def get_mask_statistics(self):
        """Get detailed statistics about mask values"""
        stats = {}
        
        for name, mask_module in self.mask_modules.items():
            mask_values = mask_module().detach().cpu()
            original_name = name.replace('_', '.')
            
            stats[original_name] = {
                'mean': mask_values.mean().item(),
                'std': mask_values.std().item(),
                'min': mask_values.min().item(),
                'max': mask_values.max().item(),
                'below_0.5': (mask_values < 0.5).sum().item(),
                'above_0.5': (mask_values >= 0.5).sum().item(),
                'total': mask_values.numel()
            }
        
        return stats
