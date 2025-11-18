import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PDDTrainer:
    """
    PDD Trainer with proper identity shortcut handling.
    
    Key insight: For blocks with identity shortcuts, conv2 mask must match 
    the input dimensions during training, not during pruning!
    """
    
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Initialize learnable masks
        self.masks = nn.ParameterDict()
        self.mask_to_layer = {}
        self.identity_constraints = {}  # Track which layers need constraints
        self._initialize_masks()
        
        # Optimizer
        params = list(student.parameters()) + list(self.masks.parameters())
        self.optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.lr_decay_epochs,
            gamma=args.lr_decay_rate
        )
        
        self.best_acc = 0.0
        self.current_epoch = 0
    
    def _initialize_masks(self):
        """
        Initialize masks with awareness of identity shortcuts.
        For conv2 in blocks WITHOUT projection shortcut, we need special handling.
        """
        mask_count = 0
        constrained_count = 0

        print("\n" + "="*70)
        print("Initializing Learnable Masks with Identity Shortcut Awareness")
        print("="*70)

        for name, module in self.student.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue

            # Add mask for every conv layer
            self._add_mask(name, module)
            mask_count += 1
            
            # Check if this is a conv2 with identity shortcut
            is_identity_constrained = False
            if 'conv2' in name and 'layer' in name:
                parts = name.split('.')
                layer_name = parts[0]
                block_idx = int(parts[1])
                
                # First block of each layer has projection shortcut
                # Other blocks have identity shortcut
                if block_idx > 0:
                    # This conv2 feeds into identity shortcut
                    # Its output channels must match input channels
                    self.identity_constraints[name] = True
                    is_identity_constrained = True
                    constrained_count += 1
            
            # Determine layer type for logging
            if name == 'conv1':
                layer_type = "Initial Conv"
            elif 'shortcut' in name:
                layer_type = "Shortcut Conv"
            elif 'conv1' in name:
                layer_type = "Block Conv1"
            elif 'conv2' in name:
                if is_identity_constrained:
                    layer_type = "Block Conv2 (Identity Constrained)"
                else:
                    layer_type = "Block Conv2"
            else:
                layer_type = "Unknown"
            
            marker = "⚠" if is_identity_constrained else "✓"
            print(f"  {marker} {name:<35} | Channels: {module.out_channels:3d} | Type: {layer_type}")

        print("="*70)
        print(f"Total Masks Created: {mask_count}")
        print(f"Identity Constrained Layers: {constrained_count}")
        print(f"Strategy: Mask all conv layers, enforce identity constraints during training")
        print("="*70 + "\n")

    def _add_mask(self, name, module):
        """Add a learnable mask for a conv layer."""
        mask = torch.rand(1, module.out_channels, 1, 1, device=self.device) * 2 - 1
        param_name = name.replace('.', '_')
        self.masks[param_name] = nn.Parameter(mask)
        self.mask_to_layer[param_name] = name
    
    def approx_sign(self, x):
        """Differentiable approximation of sign function (Equation 2)."""
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
    
    def get_constrained_mask(self, layer_name, mask):
        """
        Apply identity shortcut constraints to mask.
        For layers with identity shortcuts, we cannot prune too aggressively.
        """
        if layer_name not in self.identity_constraints:
            return mask
        
        # For identity-constrained layers, encourage keeping more channels
        # by adding a small bias towards 1
        constrained_mask = mask + 0.3  # Bias towards keeping channels
        return constrained_mask
    
    def apply_masks(self, model):
        """Apply masks to conv layers during forward pass."""
        def hook_fn(module, input, output, mask, layer_name):
            # Apply constraint if needed
            if layer_name in self.identity_constraints:
                mask = self.get_constrained_mask(layer_name, mask)
            
            binary_mask = (self.approx_sign(mask) > 0.5).float()
            return output * binary_mask

        hooks = []
        for name, module in model.named_modules():
            param_name = name.replace('.', '_')
            if param_name in self.masks:
                mask = self.masks[param_name]
                hook = module.register_forward_hook(
                    lambda m, i, o, mask=mask, name=name: hook_fn(m, i, o, mask, name)
                )
                hooks.append(hook)

        return hooks
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """KL-divergence distillation loss."""
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    def compute_mask_regularization(self):
        """
        Add L1 regularization on masks to encourage sparsity.
        This helps achieve better pruning ratios.
        """
        l1_loss = 0.0
        for mask in self.masks.values():
            # L1 on the continuous mask values
            l1_loss += torch.abs(mask).sum()
        return l1_loss * 0.0001  # Small weight for regularization
    
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
            
            # Compute losses
            distill_loss = self.distillation_loss(
                student_logits, teacher_logits, self.args.temperature
            )
            ce_loss = F.cross_entropy(student_logits, targets)
            reg_loss = self.compute_mask_regularization()
            
            # Total loss
            loss = (self.args.alpha * distill_loss + 
                    (1 - self.args.alpha) * ce_loss + 
                    reg_loss)
            
            # Backward
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
        """Calculate overall pruning ratio."""
        total_channels = 0
        pruned_channels = 0
        
        for param_name, mask in self.masks.items():
            layer_name = self.mask_to_layer[param_name]
            
            # Apply constraint if needed
            if layer_name in self.identity_constraints:
                mask = self.get_constrained_mask(layer_name, mask)
            
            binary_mask = self.approx_sign(mask)
            total_channels += binary_mask.numel()
            pruned_channels += (binary_mask < 0.5).sum().item()
        
        return pruned_channels / total_channels if total_channels > 0 else 0
    
    def get_detailed_pruning_stats(self):
        """Get per-layer pruning statistics."""
        stats = {}
        for param_name, mask in self.masks.items():
            original_name = self.mask_to_layer[param_name]
            
            # Apply constraint if needed
            if original_name in self.identity_constraints:
                mask = self.get_constrained_mask(original_name, mask)
            
            binary_mask = self.approx_sign(mask)
            total = binary_mask.numel()
            kept = (binary_mask >= 0.5).sum().item()
            pruned = total - kept
            
            is_constrained = original_name in self.identity_constraints
            
            stats[original_name] = {
                'total': total,
                'kept': kept,
                'pruned': pruned,
                'pruning_ratio': pruned / total if total > 0 else 0,
                'constrained': is_constrained
            }
        
        return stats
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("Starting Pruning During Distillation (PDD)")
        print("="*70)
        print(f"Configuration:")
        print(f"  Temperature:        {self.args.temperature}")
        print(f"  Alpha (distill):    {self.args.alpha}")
        print(f"  Learning Rate:      {self.args.lr}")
        print(f"  Epochs:             {self.args.epochs}")
        print(f"  LR Decay Epochs:    {self.args.lr_decay_epochs}")
        print(f"  LR Decay Rate:      {self.args.lr_decay_rate}")
        print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc, distill_loss, ce_loss, reg_loss = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Update LR
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Pruning stats
            pruning_ratio = self.get_pruning_ratio()
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1}/{self.args.epochs}] Summary")
            print(f"{'='*70}")
            print(f"Training:")
            print(f"  Loss:          {train_loss:.4f}")
            print(f"  Accuracy:      {train_acc:.2f}%")
            print(f"  Distill Loss:  {distill_loss:.4f}")
            print(f"  CE Loss:       {ce_loss:.4f}")
            print(f"  Reg Loss:      {reg_loss:.6f}")
            print(f"\nValidation:")
            print(f"  Loss:          {test_loss:.4f}")
            print(f"  Accuracy:      {test_acc:.2f}%")
            print(f"\nPruning:")
            print(f"  Current Ratio: {pruning_ratio*100:.2f}%")
            print(f"\nLearning Rate:")
            print(f"  Current:       {current_lr:.6f}")
            if new_lr != current_lr:
                print(f"  → Updated to:  {new_lr:.6f}")
            print(f"{'='*70}")
            
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                print(f"\n🎉 New best accuracy: {test_acc:.2f}%")
            
            # Detailed stats every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"\nDetailed Pruning Statistics (Epoch {epoch+1}):")
                print(f"{'-'*70}")
                stats = self.get_detailed_pruning_stats()
                for layer_name, stat in stats.items():
                    constraint_marker = " [CONSTRAINED]" if stat['constrained'] else ""
                    print(f"  {layer_name:<35} | "
                          f"Total: {stat['total']:3d} | "
                          f"Kept: {stat['kept']:3d} | "
                          f"Pruned: {stat['pruning_ratio']*100:5.1f}%{constraint_marker}")
                print(f"{'-'*70}")
        
        print(f"\n{'='*70}")
        print(f"Training Completed!")
        print(f"{'='*70}")
        print(f"Best Test Accuracy: {self.best_acc:.2f}%")
        print(f"Final Pruning Ratio: {self.get_pruning_ratio()*100:.2f}%")
        print(f"{'='*70}\n")
    
    def get_masks(self):
        """Return binary masks after training."""
        binary_masks = {}
        for param_name, mask in self.masks.items():
            original_name = self.mask_to_layer[param_name]
            
            # Apply constraint if needed
            if original_name in self.identity_constraints:
                mask = self.get_constrained_mask(original_name, mask)
            
            binary_masks[original_name] = self.approx_sign(mask)
        
        return binary_masks
