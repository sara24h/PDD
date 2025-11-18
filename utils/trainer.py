import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PDDTrainer:
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Initialize learnable masks for each conv layer
        # Use nn.ParameterDict to properly register masks as parameters
        self.masks = nn.ParameterDict()
        # Keep a mapping of mask names to original layer names
        self.mask_to_layer = {}
        self._initialize_masks()
        
        # Optimizer includes both model parameters and masks
        # Now masks are properly registered as parameters
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
      
        mask_count = 0
        skip_count = 0
        
        print("\n" + "="*70)
        print("Initializing Learnable Masks for Pruning During Distillation")
        print("="*70)
        
        for name, module in self.student.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            
            # Parse layer name
            parts = name.split('.')
            
            # Strategy 1: Initial conv1 - always maskable
            if name == 'conv1':
                self._add_mask(name, module)
                mask_count += 1
                print(f"  ✓ {name:<35} | Channels: {module.out_channels:3d} | Type: Initial Conv")
                continue
            
            # Strategy 2: Residual blocks in layerX
            if len(parts) >= 3 and parts[0].startswith('layer'):
                layer_name = parts[0]      # layer1, layer2, layer3
                block_idx = int(parts[1])  # 0, 1, 2
                conv_name = parts[2]       # conv1, conv2, shortcut
                
                # Rule 1: All conv1 in blocks are maskable (intermediate features)
                if conv_name == 'conv1':
                    self._add_mask(name, module)
                    mask_count += 1
                    print(f"  ✓ {name:<35} | Channels: {module.out_channels:3d} | Type: Block Conv1")
                    continue
                
                # Rule 2: conv2 only in first block (has projection shortcut)
                if conv_name == 'conv2':
                    if block_idx == 0:
                        self._add_mask(name, module)
                        mask_count += 1
                        print(f"  ✓ {name:<35} | Channels: {module.out_channels:3d} | Type: Conv2 (Proj)")
                    else:
                        skip_count += 1
                        print(f"  ✗ {name:<35} | Channels: {module.out_channels:3d} | SKIP: Identity")
                    continue
                
                # Rule 3: Shortcut conv (in block 0) - NO MASK needed
                # The shortcut will be pruned automatically based on conv2 mask
                if conv_name == 'shortcut':
                    skip_count += 1
                    print(f"  - {name:<35} | Channels: {module.out_channels:3d} | Auto: Follow Conv2")
                    continue
        
        print("="*70)
        print(f"Mask Initialization Complete:")
        print(f"  ✓ Masks created:     {mask_count}")
        print(f"  ✗ Layers skipped:    {skip_count}")
        print(f"  Strategy:            Identity shortcut constraints respected")
        print("="*70 + "\n")
    
    def _add_mask(self, name, module):

        mask = torch.rand(1, module.out_channels, 1, 1, device=self.device) * 2 - 1

        param_name = name.replace('.', '_')
        self.masks[param_name] = nn.Parameter(mask)
        
        # Keep mapping to original name
        self.mask_to_layer[param_name] = name
    
    def approx_sign(self, x):
       
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
       
        def hook_fn(module, input, output, mask):
            binary_mask = self.approx_sign(mask)
            return output * binary_mask
        
        hooks = []
        for name, module in model.named_modules():
            # Convert layer name to mask parameter name
            param_name = name.replace('.', '_')
            if param_name in self.masks:
                mask = self.masks[param_name]
                hook = module.register_forward_hook(
                    lambda m, i, o, mask=mask: hook_fn(m, i, o, mask)
                )
                hooks.append(hook)
        
        return hooks
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
    
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    def train_epoch(self):
    
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
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Student predictions with masks applied
            student_logits = self.student(inputs)
            
            # Teacher predictions (no gradient computation needed)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            # Calculate losses (Equation 4 in paper)
            # L_total = α * L_distill + (1-α) * L_CE
            distill_loss = self.distillation_loss(
                student_logits, teacher_logits, self.args.temperature
            )
            ce_loss = F.cross_entropy(student_logits, targets)
            
            # Combined loss with alpha weighting
            loss = self.args.alpha * distill_loss + (1 - self.args.alpha) * ce_loss
            
            # Backward pass - updates both model parameters and mask parameters
            loss.backward()
            self.optimizer.step()
            
            # Statistics for logging
            total_loss += loss.item()
            distill_loss_sum += distill_loss.item()
            ce_loss_sum += ce_loss.item()
            
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Remove hooks after epoch
        for hook in hooks:
            hook.remove()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_distill = distill_loss_sum / len(self.train_loader)
        avg_ce = ce_loss_sum / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return avg_loss, train_acc, avg_distill, avg_ce
    
    def evaluate(self):
       
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
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def get_pruning_ratio(self):
       
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            binary_mask = self.approx_sign(mask)
            total_channels += binary_mask.numel()
            pruned_channels += (binary_mask < 0.5).sum().item()
        
        return pruned_channels / total_channels if total_channels > 0 else 0
    
    def get_detailed_pruning_stats(self):
      
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
            
            # Evaluate on test set
            test_loss, test_acc = self.evaluate()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Get pruning statistics
            pruning_ratio = self.get_pruning_ratio()
            
            # Print statistics
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
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                print(f"\n🎉 New best accuracy: {test_acc:.2f}%")
            
            # Print detailed pruning stats every 10 epochs
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
       
        binary_masks = {}
        for param_name, mask in self.masks.items():
            # Convert back to original layer name
            original_name = self.mask_to_layer[param_name]
            binary_masks[original_name] = self.approx_sign(mask)
        return binary_masks
