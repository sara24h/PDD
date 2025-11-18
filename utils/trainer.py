# ============================================================================
# File: utils/trainer.py
# Fixed PDD Trainer - Matches Paper Methodology
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PDDTrainer:
    """
    PDD Trainer that matches the paper's methodology exactly.
    
    Key fixes:
    1. All conv layers get independent masks (no constraints during training)
    2. Masks are applied ONLY to output channels
    3. Identity shortcut issues are handled during pruning, not training
    4. Simple L1 regularization on masks for sparsity
    """
    
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Initialize learnable masks for ALL conv layers
        self.masks = nn.ParameterDict()
        self.mask_to_layer = {}
        self._initialize_masks()
        
        # Optimizer includes both model and mask parameters
        params = list(student.parameters()) + list(self.masks.parameters())
        self.optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.lr_decay_epochs,
            gamma=args.lr_decay_rate
        )
        
        self.best_acc = 0.0
        self.current_epoch = 0
    
    def _initialize_masks(self):
        """Initialize learnable masks for all conv layers."""
        mask_count = 0
        
        print("\n" + "="*70)
        print("Initializing Learnable Masks (Paper Methodology)")
        print("="*70)
        
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                # Initialize mask randomly in range [-1, 1]
                mask = torch.rand(1, module.out_channels, 1, 1, device=self.device) * 2 - 1
                
                # Store mask as learnable parameter
                param_name = name.replace('.', '_')
                self.masks[param_name] = nn.Parameter(mask)
                self.mask_to_layer[param_name] = name
                
                mask_count += 1
                print(f"  ✓ {name:<35} | Channels: {module.out_channels:3d} | Mask shape: {mask.shape}")
        
        print("="*70)
        print(f"Total Masks: {mask_count}")
        print("All conv layers will be pruned independently")
        print("="*70 + "\n")
    
    def approx_sign(self, x):
        """
        Differentiable piecewise polynomial function (Equation 2 from paper).
        This approximates the sign function with smooth gradients.
        """
        return torch.where(
            x < -1,
            torch.zeros_like(x),
            torch.where(
                x < 0,
                (x + 1) ** 2 / 2,
                torch.where(
                    x < 1,
                    (2 * x - x ** 2 + 1) / 2,
                    torch.ones_like(x)
                )
            )
        )
    
    def apply_masks(self, model):
        """
        Apply masks to conv layer outputs during forward pass.
        This implements the dynamic mask from Equation (3) in the paper.
        """
        hooks = []
        
        def create_hook(mask):
            def hook_fn(module, input, output):
                # Apply differentiable sign approximation
                binary_mask = self.approx_sign(mask)
                # Multiply output by mask
                return output * binary_mask
            return hook_fn
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                param_name = name.replace('.', '_')
                if param_name in self.masks:
                    mask = self.masks[param_name]
                    hook = module.register_forward_hook(create_hook(mask))
                    hooks.append(hook)
        
        return hooks
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        KL-divergence distillation loss.
        This is the L(z_s, z_t) term from the paper.
        """
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kl_loss * (temperature ** 2)
    
    def mask_regularization(self):
        """
        L1 regularization on masks to encourage sparsity.
        Helps achieve better pruning ratios.
        """
        l1_loss = 0.0
        for mask in self.masks.values():
            l1_loss += torch.abs(mask).sum()
        return l1_loss * 1e-4
    
    def train_epoch(self):
        """Train for one epoch with mask learning."""
        self.student.train()
        hooks = self.apply_masks(self.student)
        
        total_loss = 0
        distill_loss_sum = 0
        ce_loss_sum = 0
        reg_loss_sum = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.args.epochs}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            student_logits = self.student(inputs)
            
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            # Compute losses (Equation 4 from paper)
            distill_loss = self.distillation_loss(
                student_logits, teacher_logits, self.args.temperature
            )
            ce_loss = F.cross_entropy(student_logits, targets)
            reg_loss = self.mask_regularization()
            
            # Total loss
            loss = (self.args.alpha * distill_loss + 
                    (1 - self.args.alpha) * ce_loss + 
                    reg_loss)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            distill_loss_sum += distill_loss.item()
            ce_loss_sum += ce_loss.item()
            reg_loss_sum += reg_loss.item()
            
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_distill = distill_loss_sum / len(self.train_loader)
        avg_ce = ce_loss_sum / len(self.train_loader)
        avg_reg = reg_loss_sum / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return avg_loss, train_acc, avg_distill, avg_ce, avg_reg
    
    def evaluate(self):
        """Evaluate on test set."""
        self.student.eval()
        hooks = self.apply_masks(self.student)
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.student(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        for hook in hooks:
            hook.remove()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def get_pruning_ratio(self):
        """Calculate current pruning ratio based on masks."""
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            binary_mask = self.approx_sign(mask)
            total_channels += binary_mask.numel()
            pruned_channels += (binary_mask < 0.5).sum().item()
        
        return pruned_channels / total_channels if total_channels > 0 else 0
    
    def train(self):
        """Main training loop."""
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
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc, distill_loss, ce_loss, reg_loss = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Get pruning ratio
            pruning_ratio = self.get_pruning_ratio()
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1}/{self.args.epochs}]")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print(f"Distill: {distill_loss:.4f} | CE: {ce_loss:.4f} | Reg: {reg_loss:.6f}")
            print(f"Pruning Ratio: {pruning_ratio*100:.2f}%")
            if new_lr != old_lr:
                print(f"LR: {old_lr:.6f} → {new_lr:.6f}")
            print(f"{'='*70}")
            
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                print(f"🎉 New best accuracy: {test_acc:.2f}%")
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        print(f"Final Pruning: {self.get_pruning_ratio()*100:.2f}%")
        print(f"{'='*70}\n")
    
    def get_masks(self):
        """Return binary masks after training."""
        binary_masks = {}
        for param_name, mask in self.masks.items():
            layer_name = self.mask_to_layer[param_name]
            binary_masks[layer_name] = self.approx_sign(mask).detach()
        return binary_masks
