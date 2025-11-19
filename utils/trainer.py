import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # Optimizer - only model parameters + masks
        mask_params = [mask for mask in self.masks.values()]
        
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
        """Initialize learnable masks for each convolutional layer"""
        masks = {}
        
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                # Initialize with random normal distribution (paper uses this)
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device),
                    requires_grad=True
                )
                masks[name] = mask
        
        return masks

    def _approx_sign(self, x):
        """Differentiable approximation of sign function from paper (Equation 2)
        
        ApproxSign(x) = {
            0                   if x < -1
            (x+1)²/2           if -1 ≤ x < 0
            (2x - x² + 1)/2    if 0 ≤ x < 1
            1                   if x ≥ 1
        }
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

    def train(self):
        """Train the student model with pruning during distillation"""
        print("\n" + "="*70)
        print("Starting Pruning During Distillation (PDD) - Paper Implementation")
        print("="*70)
        print(f"Temperature:        {self.args.temperature}")
        print(f"Alpha (distill):    {self.args.alpha}")
        print(f"Learning Rate:      {self.args.lr}")
        print(f"Epochs:             {self.args.epochs}")
        print(f"LR Decay:           {self.args.lr_decay_epochs}")
        print(f"Total Masks:        {len(self.masks)}")
        print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            # Train
            self.student.train()
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # ⭐ CRITICAL FIX: Don't apply masks during training!
                # Masks are only used to identify channels for pruning AFTER training
                student_outputs = self.student(inputs)
                
                # Teacher outputs (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs)
                
                # Calculate losses (Equation 4 from paper)
                # 1. Classification loss
                ce_loss = self.criterion(student_outputs, targets)
                
                # 2. Knowledge distillation loss
                soft_teacher = F.softmax(teacher_outputs / self.args.temperature, dim=1)
                soft_student = F.log_softmax(student_outputs / self.args.temperature, dim=1)
                kd_loss = self.kd_criterion(soft_student, soft_teacher) * (self.args.temperature ** 2)
                
                # ⭐ CRITICAL FIX: Total loss WITHOUT regularization (as per paper Eq. 4)
                # L_total = L(z_s, z_t) + CE(z_s, Y)
                total_loss = self.args.alpha * kd_loss + (1 - self.args.alpha) * ce_loss
                
                # Backward pass
                total_loss.backward()
                
                # ⭐ IMPORTANT: Masks are updated through gradient descent
                # They will naturally converge to identify redundant channels
                self.optimizer.step()
                
                # Track statistics
                train_loss += total_loss.item()
                kd_loss_total += kd_loss.item()
                ce_loss_total += ce_loss.item()
                
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            
            # Evaluate
            test_acc = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()

            # Calculate current pruning ratio for monitoring
            pruning_ratio = self._calculate_pruning_ratio()
            
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train: Loss={train_loss/len(self.train_loader):.4f}, Acc={train_acc:.2f}%")
            print(f"Test:  Acc={test_acc:.2f}%")
            print(f"Losses: KD={kd_loss_total/len(self.train_loader):.4f}, "
                  f"CE={ce_loss_total/len(self.train_loader):.4f}")
            print(f"Estimated Pruning Ratio: {pruning_ratio:.2f}%")
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_masks = {name: mask.clone().detach() 
                                  for name, mask in self.masks.items()}
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        # Restore best masks
        if self.best_masks is not None:
            for name in self.masks.keys():
                self.masks[name].data = self.best_masks[name].data
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        print(f"Final Estimated Pruning: {self._calculate_pruning_ratio():.2f}%")
        print("="*70 + "\n")

    def evaluate(self):
        """Evaluate the student model WITHOUT masks"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # ⭐ Evaluate without mask application
                outputs = self.student(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total

    def _calculate_pruning_ratio(self):
        """Calculate the current pruning ratio based on mask values"""
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            binary_mask = self._approx_sign(mask)
            total_channels += mask.numel()
            # Channels with score < 0.5 will be pruned
            pruned_channels += (binary_mask < 0.5).sum().item()
        
        if total_channels == 0:
            return 0.0
        
        return 100. * pruned_channels / total_channels

    def get_masks(self):
        """Get the binary masks for pruning (threshold at 0.5)"""
        binary_masks = {}
        for name, mask in self.masks.items():
            # Apply ApproxSign
            continuous_mask = self._approx_sign(mask)
            # ⭐ Threshold at 0.5 (as per paper)
            binary_masks[name] = (continuous_mask > 0.5).float()
        return binary_masks
