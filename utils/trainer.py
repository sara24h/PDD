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
        
        # Initialize masks
        self.masks = self._initialize_masks()
        
        # Optimizer
        self.optimizer = torch.optim.SGD(
            list(self.student.parameters()) + list(self.masks.values()),
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
            # Initialize mask with shape [1, out_channels, 1, 1]
                mask = nn.Parameter(torch.randn(1, module.out_channels, 1, 1))
                mask = mask.to(self.device)
                masks[name] = mask

            elif 'shortcut' in name and isinstance(module, nn.Sequential):
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, nn.Conv2d):
                        mask = nn.Parameter(torch.randn(1, sub_module.out_channels, 1, 1))
                        mask = mask.to(self.device)
                        masks[f"{name}.{sub_name}"] = mask
    
        return masks

    def _apply_masks(self):
        """Apply masks to the student model"""
        for name, module in self.student.named_modules():
            if name in self.masks and isinstance(module, nn.Conv2d):
             # Apply the ApproxSign function to the mask
                mask = self._approx_sign(self.masks[name])

                weight_mask = mask.squeeze(0).view(-1, 1, 1, 1)
                module.weight.data *= weight_mask

    def _approx_sign(self, x):
        """Differentiable approximation of sign function from the paper"""
        result = torch.zeros_like(x)
        
        # x < -1
        mask1 = (x < -1).float()
        result += mask1 * 0.0
        
        # -1 <= x < 0
        mask2 = ((x >= -1) & (x < 0)).float()
        result += mask2 * ((x + 1) ** 2 / 2)
        
        # 0 <= x < 1
        mask3 = ((x >= 0) & (x < 1)).float()
        result += mask3 * (2 * x - x**2 + 1) / 2
        
        # x >= 1
        mask4 = (x >= 1).float()
        result += mask4 * 1.0
        
        return result

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
        print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            # Train
            self.student.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Apply masks to the model
                self._apply_masks()
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Student outputs
                student_outputs = self.student(inputs)
                
                # Teacher outputs (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs)
                
                # Calculate losses
                ce_loss = self.criterion(student_outputs, targets)
                
                # Knowledge distillation loss
                soft_teacher = F.softmax(teacher_outputs / self.args.temperature, dim=1)
                soft_student = F.log_softmax(student_outputs / self.args.temperature, dim=1)
                kd_loss = self.kd_criterion(soft_student, soft_teacher) * (self.args.temperature ** 2)
                
                # Regularization loss to encourage pruning
                reg_loss = 0.0
                for mask in self.masks.values():
                    # L1 regularization on masks to encourage sparsity
                    reg_loss += torch.mean(torch.abs(self._approx_sign(mask)))
                
                # Total loss
                total_loss = self.args.alpha * kd_loss + (1 - self.args.alpha) * ce_loss + 0.0001 * reg_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += total_loss.item()
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Evaluate
            test_acc = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate pruning ratio
            pruning_ratio = self._calculate_pruning_ratio()
            
            # Print statistics
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train Loss: {train_loss/len(self.train_loader):.4f} | Train Acc: {100.*correct/total:.2f}%")
            print(f"Test Loss:  {self._get_test_loss():.4f} | Test Acc:  {test_acc:.2f}%")
            print(f"Distill: {kd_loss.item():.4f} | CE: {ce_loss.item():.4f} | Reg: {reg_loss.item():.6f}")
            print(f"Pruning Ratio: {pruning_ratio:.2f}%")
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_masks = {name: mask.clone().detach() for name, mask in self.masks.items()}
                print(f"🎉 New best accuracy: {test_acc:.2f}%")
        
        # Restore best masks
        if self.best_masks is not None:
            self.masks = self.best_masks
        
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
                
                # Apply masks
                self._apply_masks()
                
                outputs = self.student(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total

    def _get_test_loss(self):
        """Get test loss"""
        self.student.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Apply masks
                self._apply_masks()
                
                outputs = self.student(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
        
        return test_loss / len(self.test_loader)

    def _calculate_pruning_ratio(self):
        """Calculate the current pruning ratio"""
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            binary_mask = self._approx_sign(mask)
            total_channels += mask.numel()
            pruned_channels += (binary_mask == 0).sum().item()
        
        return 100. * pruned_channels / total_channels

    def get_masks(self):
        """Get the current masks"""
        return self.masks
