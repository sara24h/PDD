import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PDDTrainer:
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Initialize masks as nn.Parameters
        self.masks = self._initialize_masks()
        
        # Optimizer - بهینه‌سازی پارامترهای مدل و ماسک‌ها
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
        Initialize learnable masks for each convolutional layer
        
        Paper: "First of all, we randomly initialized each mask x_i"
        No special scaling mentioned - just standard normal distribution
        """
        masks = {}
        
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                # مطابق مقاله: یک ماسک با شکل [1, out_channels, 1, 1]
                # مقداردهی اولیه تصادفی با توزیع نرمال (بدون scaling)
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device),
                    requires_grad=True
                )
                masks[name] = mask
        
        return masks

    def _approx_sign(self, x):
        """
        Differentiable approximation of sign function from the paper (Equation 2)
        
        ApproxSign(x) = {
            0                    if x < -1
            (x+1)²/2            if -1 ≤ x < 0
            (2x - x² + 1)/2     if 0 ≤ x < 1
            1                    otherwise
        }
        
        This function maps raw mask values to [0, 1] range
        Paper: "a score of 0 indicates that the channel is redundant and can be pruned"
        """
        result = torch.zeros_like(x)
        
        # x < -1: output = 0
        mask1 = (x < -1).float()
        result = result + mask1 * 0.0
        
        # -1 ≤ x < 0: output = (x+1)²/2
        mask2 = ((x >= -1) & (x < 0)).float()
        result = result + mask2 * ((x + 1) ** 2 / 2)
        
        # 0 ≤ x < 1: output = (2x - x² + 1)/2
        mask3 = ((x >= 0) & (x < 1)).float()
        result = result + mask3 * (2 * x - x**2 + 1) / 2
        
        # x ≥ 1: output = 1
        mask4 = (x >= 1).float()
        result = result + mask4 * 1.0
        
        return result

    def _forward_with_masks(self, x):
        """
        Forward pass with mask application (Equation 3 from paper)
        
        The paper defines: z_s = h_n(h_{n-1}(...h_0(M)·A(x_0)...)·A(x_{n-1}))·A(x_n)
        where A(·) is the ApproxSign function
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
        """Train the student model with pruning during distillation"""
        print("\n" + "="*70)
        print("Starting Pruning During Distillation (PDD)")
        print("="*70)
        print(f"Temperature:        {self.args.temperature}")
        print(f"Alpha (distill):    {self.args.alpha}")
        print(f"Learning Rate:      {self.args.lr}")
        print(f"Epochs:             {self.args.epochs}")
        print(f"LR Decay:           {self.args.lr_decay_epochs}")
        print(f"Total Masks:        {len(self.masks)}")
        print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            # Training phase
            self.student.train()
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            #reg_loss_total = 0.0
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
                
                # Calculate losses (Equation 4)
                
                # 1. Classification loss (Cross Entropy)
                ce_loss = self.criterion(student_outputs, targets)
                
                # 2. Knowledge distillation loss (KL Divergence)
                soft_teacher = F.softmax(teacher_outputs / self.args.temperature, dim=1)
                soft_student = F.log_softmax(student_outputs / self.args.temperature, dim=1)
                kd_loss = self.kd_criterion(soft_student, soft_teacher) * (self.args.temperature ** 2)
                
                # 3. Sparsity regularization (L1 on binary masks)
                # This encourages masks to be either 0 (pruned) or 1 (kept)
                #reg_loss = 0.0
                total_mask_elements = 0
                for mask in self.masks.values():
                    # Apply ApproxSign to get binary-like values [0, 1]
                    binary_mask = self._approx_sign(mask)
                    # L1 regularization: encourages sparsity
                    #reg_loss += torch.sum(binary_mask)
                    total_mask_elements += binary_mask.numel()
                
                # Normalize regularization by total number of mask elements
                #if total_mask_elements > 0:
                    #reg_loss = reg_loss / total_mask_elements
                
                # Total loss (Equation 4: L_total = L(z_s, z_t) + CE(z_s, Y))
                # Added regularization term with weight 0.1
                total_loss = (self.args.alpha * kd_loss + 
                             (1 - self.args.alpha) * ce_loss 
                
                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += total_loss.item()
                kd_loss_total += kd_loss.item()
                ce_loss_total += ce_loss.item()
                #reg_loss_total += reg_loss.item()
                
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            
            # Evaluation phase
            test_acc = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()

            # Calculate current pruning ratio (using threshold=0 as per paper)
            pruning_ratio = self._calculate_pruning_ratio(threshold=0.0)
            
            # Print epoch statistics
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train: Loss={train_loss/len(self.train_loader):.4f}, Acc={train_acc:.2f}%")
            print(f"Test:  Acc={test_acc:.2f}%")
            print(f"Losses: KD={kd_loss_total/len(self.train_loader):.4f}, "
                  f"CE={ce_loss_total/len(self.train_loader):.4f}, "
                  #f"Reg={reg_loss_total/len(self.train_loader):.6f}")
            print(f"Pruning Ratio (score=0): {pruning_ratio:.2f}%")
            
            # Save best model based on test accuracy
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                # Clone masks for best model
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
        print(f"Final Pruning (score=0): {self._calculate_pruning_ratio(threshold=0.0):.2f}%")
        print("="*70 + "\n")

    def evaluate(self):
        """Evaluate the student model on test set"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass with masks
                outputs = self._forward_with_masks(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total

    def _calculate_pruning_ratio(self, threshold=0.0):
        
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            # Apply ApproxSign to get values in [0, 1]
            binary_mask = self._approx_sign(mask)
            total_channels += binary_mask.numel()
            # Count channels at or below threshold as pruned
            pruned_channels += (binary_mask <= threshold).sum().item()
        
        if total_channels == 0:
            return 0.0
        
        return 100. * pruned_channels / total_channels

    def get_masks(self, threshold=0.0):
      
        binary_masks = {}
        for name, mask in self.masks.items():
            # Apply ApproxSign to get values in [0, 1]
            binary_mask = self._approx_sign(mask)
            # Convert to strict binary: 0 if <= threshold, 1 if > threshold
            binary_masks[name] = (binary_mask > threshold).float()
        return binary_masks

    def get_mask_statistics(self):
        """Get detailed statistics about current mask values"""
        stats = {}
        
        for name, mask in self.masks.items():
            # Raw mask values
            raw_values = mask.detach().cpu()
            # After ApproxSign
            approx_values = self._approx_sign(mask).detach().cpu()
            
            stats[name] = {
                'raw_mean': raw_values.mean().item(),
                'raw_std': raw_values.std().item(),
                'raw_min': raw_values.min().item(),
                'raw_max': raw_values.max().item(),
                'approx_mean': approx_values.mean().item(),
                'approx_std': approx_values.std().item(),
                'approx_min': approx_values.min().item(),
                'approx_max': approx_values.max().item(),
                'exactly_zero': (approx_values == 0.0).sum().item(),
                'above_zero': (approx_values > 0.0).sum().item(),
                'total': approx_values.numel()
            }
        
        return stats
