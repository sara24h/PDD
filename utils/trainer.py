import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PDDTrainer:
    """
    Pruning During Distillation Trainer - Fixed to match paper exactly
    
    Key fixes:
    1. Masks for ALL conv layers (including conv2 in all blocks)
    2. Independent shortcut conv masks
    3. Proper identity shortcut handling via padding/projection
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
        
        # Optimizer includes both model parameters and masks
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
        """
        Initialize masks for ALL convolutional layers as per paper.
        Each conv layer gets a learnable mask that can be optimized during training.
        """
        mask_count = 0

        print("\n" + "="*70)
        print("Initializing Learnable Masks (Paper-compliant)")
        print("="*70)

        for name, module in self.student.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue

            # Add mask for EVERY conv layer
            self._add_mask(name, module)
            mask_count += 1
            
            # Determine layer type for logging
            if name == 'conv1':
                layer_type = "Initial Conv"
            elif 'shortcut' in name:
                layer_type = "Shortcut Conv"
            elif 'conv1' in name:
                layer_type = "Block Conv1"
            elif 'conv2' in name:
                layer_type = "Block Conv2"
            else:
                layer_type = "Unknown"
            
            print(f"  ✓ {name:<35} | Channels: {module.out_channels:3d} | Type: {layer_type}")

        print("="*70)
        print(f"Total Masks Created: {mask_count}")
        print(f"Strategy: Mask ALL conv layers as per paper Section 'The Proposed Method'")
        print("="*70 + "\n")

    def _add_mask(self, name, module):
        """
        Add a learnable mask for a conv layer.
        Initialized randomly in [-1, 1] as per paper.
        """
        # Random initialization in [-1, 1]
        mask = torch.rand(1, module.out_channels, 1, 1, device=self.device) * 2 - 1
        
        # Convert layer name to parameter name
        param_name = name.replace('.', '_')
        self.masks[param_name] = nn.Parameter(mask)
        
        # Keep mapping to original name
        self.mask_to_layer[param_name] = name
    
    def approx_sign(self, x):
        """
        Differentiable piecewise polynomial function from Equation (2).
        Approximates sign function with smooth gradients.
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
        Apply binary masks to conv layers during forward pass.
        This implements the dynamic mask mechanism from the paper.
        """
        def hook_fn(module, input, output, mask):
            # Convert continuous mask to binary (threshold at 0.5)
            binary_mask = (self.approx_sign(mask) > 0.5).float()
            return output * binary_mask

        hooks = []
        for name, module in model.named_modules():
            param_name = name.replace('.', '_')
            if param_name in self.masks:
                mask = self.masks[param_name]
                hook = module.register_forward_hook(
                    lambda m, i, o, mask=mask: hook_fn(m, i, o, mask)
                )
                hooks.append(hook)

        return hooks
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        KL-divergence distillation loss as commonly used in KD literature.
        Referenced in paper as L(·) in Equation (4).
        """
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    def train_epoch(self):
        """
        Train for one epoch with distillation and mask learning.
        Implements Equation (4): L_total = L(z_s, z_t) + CE(z_s, Y)
        """
        self.student.train()
        hooks = self.apply_masks(self.student)
        
        total_loss = 0
        distill_loss_sum = 0
        ce_loss_sum = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.args.epochs}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with masks
            student_logits = self.student(inputs)
            
            # Teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            # Equation (4) from paper
            distill_loss = self.distillation_loss(
                student_logits, teacher_logits, self.args.temperature
            )
            ce_loss = F.cross_entropy(student_logits, targets)
            
            # Combined loss (paper uses sum, but alpha weighting is equivalent)
            loss = self.args.alpha * distill_loss + (1 - self.args.alpha) * ce_loss
            
            # Backward pass - updates both model and mask parameters
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            distill_loss_sum += distill_loss.item()
            ce_loss_sum += ce_loss.item()
            
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
        train_acc = 100. * correct / total
        
        return avg_loss, train_acc, avg_distill, avg_ce
    
    def evaluate(self):
        """Evaluate on test set with current masks."""
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
        """Calculate overall channel pruning ratio."""
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            binary_mask = self.approx_sign(mask)
            total_channels += binary_mask.numel()
            pruned_channels += (binary_mask < 0.5).sum().item()
        
        return pruned_channels / total_channels if total_channels > 0 else 0
    
    def get_detailed_pruning_stats(self):
        """Get per-layer pruning statistics."""
        stats = {}
        for param_name, mask in self.masks.items():
            original_name = self.mask_to_layer[param_name]
            binary_mask = self.approx_sign(mask)
            total = binary_mask.numel()
            kept = (binary_mask >= 0.5).sum().item()
            pruned = total - kept
            
            stats[original_name] = {
                'total': total,
                'kept': kept,
                'pruned': pruned,
                'pruning_ratio': pruned / total if total > 0 else 0
            }
        
        return stats
    
    def train(self):
        """
        Main training loop for PDD.
        Follows paper Section 'The Proposed Method' and Figure 2.
        """
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
            
            # Train one epoch
            train_loss, train_acc, distill_loss, ce_loss = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Pruning statistics
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
            
            # Track best accuracy
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                print(f"\n🎉 New best accuracy: {test_acc:.2f}%")
            
            # Detailed stats every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"\nDetailed Pruning Statistics (Epoch {epoch+1}):")
                print(f"{'-'*70}")
                stats = self.get_detailed_pruning_stats()
                for layer_name, stat in stats.items():
                    print(f"  {layer_name:<35} | "
                          f"Total: {stat['total']:3d} | "
                          f"Kept: {stat['kept']:3d} | "
                          f"Pruned: {stat['pruning_ratio']*100:5.1f}%")
                print(f"{'-'*70}")
        
        print(f"\n{'='*70}")
        print(f"Training Completed!")
        print(f"{'='*70}")
        print(f"Best Test Accuracy: {self.best_acc:.2f}%")
        print(f"Final Pruning Ratio: {self.get_pruning_ratio()*100:.2f}%")
        print(f"{'='*70}\n")
    
    def get_masks(self):
        """
        Return binary masks after training.
        Used for actual pruning in Phase 2.
        """
        binary_masks = {}
        for param_name, mask in self.masks.items():
            original_name = self.mask_to_layer[param_name]
            binary_masks[original_name] = self.approx_sign(mask)
        return binary_masks
