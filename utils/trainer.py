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
        self.masks = {}
        self._initialize_masks()
        
        # Optimizer includes both model parameters and masks
        params = list(student.parameters()) + list(self.masks.values())
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
  
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
            # Random initialization around 0 (uniform distribution [-1, 1])
            # This gives approximately 50% chance of each channel being kept/pruned
                mask = nn.Parameter(torch.rand(1, module.out_channels, 1, 1) * 2 - 1)
            # تغییر اینجا: مستقیماً تنسور را به دستگاه مورد نظر منتقل کنید
                self.masks[name] = mask.to(self.device) # <-- اینطوری درست می‌شود
        
        # اگر دستگاه مدل تغییر کند، باید ماسک‌ها هم تغییر کنند. پس ممکن است بخواهید اطمینان حاصل کنید
        # که ماسک‌ها همیشه روی همان دستگاهی هستند که مدل است. اما در این حالت، دستگاه ثابت است.
        
    print(f"Initialized {len(self.masks)} learnable masks")
    
    def approx_sign(self, x):
        """
        Differentiable piecewise polynomial function (Equation 2 in paper)
        
        ApproxSign(x) = {
            0              if x < -1
            (x+1)²/2       if -1 ≤ x < 0
            (2x - x² + 1)/2 if 0 ≤ x < 1
            1              otherwise
        }
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
        """Apply binarized masks to model's forward pass"""
        def hook_fn(module, input, output, mask):
            binary_mask = self.approx_sign(mask)
            return output * binary_mask
        
        hooks = []
        for name, module in model.named_modules():
            if name in self.masks:
                mask = self.masks[name]
                hook = module.register_forward_hook(
                    lambda m, i, o, mask=mask: hook_fn(m, i, o, mask)
                )
                hooks.append(hook)
        
        return hooks
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        KL Divergence loss for knowledge distillation
        L(z̃_s, z_t) in equation (4)
        """
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    def train_epoch(self):
        """Train for one epoch"""
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
            
            # Student predictions with masks
            student_logits = self.student(inputs)
            
            # Teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            # Calculate losses (Equation 4 in paper)
            distill_loss = self.distillation_loss(
                student_logits, teacher_logits, self.args.temperature
            )
            ce_loss = F.cross_entropy(student_logits, targets)
            
            # Total loss: L_total = L(z̃_s, z_t) + CE(z̃_s, Y)
            # Paper doesn't specify weights, so we use alpha to balance
            loss = self.args.alpha * distill_loss + (1 - self.args.alpha) * ce_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
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
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_distill = distill_loss_sum / len(self.train_loader)
        avg_ce = ce_loss_sum / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return avg_loss, train_acc, avg_distill, avg_ce
    
    def evaluate(self):
        """Evaluate model"""
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
        """Calculate current pruning ratio"""
        total_channels = 0
        pruned_channels = 0
        
        for mask in self.masks.values():
            binary_mask = self.approx_sign(mask)
            total_channels += binary_mask.numel()
            pruned_channels += (binary_mask < 0.5).sum().item()
        
        return pruned_channels / total_channels if total_channels > 0 else 0
    
    def train(self):
        """Main training loop"""
        print("\nStarting Pruning During Distillation...")
        print(f"Temperature: {self.args.temperature}, Alpha: {self.args.alpha}")
        
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc, distill_loss, ce_loss = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Get pruning statistics
            pruning_ratio = self.get_pruning_ratio()
            
            # Print statistics
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print(f"Distill Loss: {distill_loss:.4f} | CE Loss: {ce_loss:.4f}")
            print(f"Current Pruning Ratio: {pruning_ratio*100:.2f}%")
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                print(f"New best accuracy: {test_acc:.2f}%")
        
        print(f"\nTraining completed!")
        print(f"Best Test Accuracy: {self.best_acc:.2f}%")
    
    def get_masks(self):
        """Return binarized masks"""
        binary_masks = {}
        for name, mask in self.masks.items():
            binary_masks[name] = self.approx_sign(mask)
        return binary_masks
