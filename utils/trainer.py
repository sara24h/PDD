import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class ApproxSign(torch.autograd.Function):
    """
    Differentiable approximation of sign function
    Using piecewise polynomial function from the paper
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Binarize the mask
        output = torch.zeros_like(x)
        output[x < -1] = 0
        mask1 = (x >= -1) & (x < 0)
        output[mask1] = ((x[mask1] + 1) ** 2) / 2
        mask2 = (x >= 0) & (x < 1)
        output[mask2] = (2 * x[mask2] - x[mask2] ** 2 + 1) / 2
        output[x >= 1] = 1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # Triangle-shaped derivative
        grad_mask = torch.zeros_like(x)
        mask1 = (x >= -1) & (x < 0)
        grad_mask[mask1] = x[mask1] + 1
        mask2 = (x >= 0) & (x < 1)
        grad_mask[mask2] = 1 - x[mask2]
        
        return grad_input * grad_mask


class PDDTrainer:
    """
    Trainer for Pruning During Distillation
    """
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # Initialize masks for each conv layer
        self.masks = {}
        self._initialize_masks()
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.SGD(
            list(self.student.parameters()) + list(self.masks.values()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.lr_decay_epochs,
            gamma=args.lr_decay_rate
        )
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()
        
    def _initialize_masks(self):
        """Initialize differentiable masks for each conv layer"""
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                # Initialize mask with random values
                mask = torch.randn(1, module.out_channels, 1, 1, 
                                  requires_grad=True, device=self.device)
                self.masks[name] = mask
    
    def _apply_masks(self):
        """Apply masks to conv layers"""
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.masks:
                mask = ApproxSign.apply(self.masks[name])
                # Register hook to apply mask during forward pass
                def hook(module, input, output, mask=mask):
                    return output * mask
                module.register_forward_hook(hook)
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        Compute KL divergence loss for distillation
        """
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kl_loss * (temperature ** 2)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.student.train()
        self.teacher.eval()
        
        total_loss = 0.0
        distill_loss_sum = 0.0
        ce_loss_sum = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply masks before forward pass
            self._apply_masks()
            
            # Forward pass
            student_logits = self.student(inputs)
            
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            # Compute losses
            ce_loss = self.ce_loss(student_logits, targets)
            distill_loss = self.distillation_loss(
                student_logits, teacher_logits, self.args.temperature
            )
            
            # Total loss
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
        
        avg_loss = total_loss / len(self.train_loader)
        avg_distill = distill_loss_sum / len(self.train_loader)
        avg_ce = ce_loss_sum / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return avg_loss, avg_distill, avg_ce, train_acc
    
    def evaluate(self):
        """Evaluate the student model"""
        self.student.eval()
        
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Apply masks
                self._apply_masks()
                
                outputs = self.student(inputs)
                loss = self.ce_loss(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        avg_loss = test_loss / len(self.test_loader)
        
        return avg_loss, test_acc
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting Pruning During Distillation...")
        print(f"Temperature: {self.args.temperature}, Alpha: {self.args.alpha}")
        
        best_acc = 0.0
        
        for epoch in range(self.args.epochs):
            # Train
            train_loss, distill_loss, ce_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Print statistics
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print(f"Distill Loss: {distill_loss:.4f} | CE Loss: {ce_loss:.4f}")
            
            # Count non-zero channels
            total_channels = 0
            kept_channels = 0
            for name, mask in self.masks.items():
                binary_mask = ApproxSign.apply(mask)
                total_channels += binary_mask.numel()
                kept_channels += (binary_mask > 0.5).sum().item()
            
            pruning_ratio = (1 - kept_channels / total_channels) * 100
            print(f"Current Pruning Ratio: {pruning_ratio:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                print(f"New best accuracy: {best_acc:.2f}%")
        
        print(f"\nTraining completed!")
        print(f"Best Test Accuracy: {best_acc:.2f}%")
        
        return best_acc
    
    def get_masks(self):
        """Get binary masks after training"""
        binary_masks = {}
        for name, mask in self.masks.items():
            binary_mask = ApproxSign.apply(mask)
            binary_masks[name] = (binary_mask > 0.5).cpu()
        return binary_masks
