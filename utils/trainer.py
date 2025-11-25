import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PDDTrainer:
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args

        self.masks = self._initialize_masks()
        mask_params = list(self.masks.values())
        
        self.optimizer = torch.optim.SGD(
            [
                {'params': self.student.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': mask_params, 'lr': args.lr * 5.0, 'weight_decay': 0.0}
            ],
            momentum=args.momentum
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate
        )

        self.criterion = nn.BCEWithLogitsLoss()  # ✅ BCE
        # No KLDivLoss — we use MSE on logits
        
        self.best_acc = 0.0
        self.best_masks = None

    def _initialize_masks(self):
        masks = {}
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = nn.Parameter(
                    torch.full((1, module.out_channels, 1, 1), -0.5, device=self.device),
                    requires_grad=True
                )
                masks[name] = mask
        return masks

    def _approx_sign(self, x):
        return torch.where(x < -1, torch.zeros_like(x),
                           torch.where(x < 0, (x + 1)**2 / 2,
                                       torch.where(x < 1, (2*x - x**2 + 1)/2, 
                                                   torch.ones_like(x))))

    def _forward_with_masks(self, x):
        out = self.student.conv1(x)
        if 'conv1' in self.masks:
            mask = self._approx_sign(self.masks['conv1'])
            out = out * mask
        out = self.student.bn1(out)
        out = F.relu(out)
        
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.student, layer_name)
            for i, block in enumerate(layer):
                identity = out
                
                out = block.conv1(out)
                mask_name = f'{layer_name}.{i}.conv1'
                if mask_name in self.masks:
                    mask = self._approx_sign(self.masks[mask_name])
                    out = out * mask
                out = block.bn1(out)
                out = F.relu(out)
                
                out = block.conv2(out)
                mask_name = f'{layer_name}.{i}.conv2'
                if mask_name in self.masks:
                    mask = self._approx_sign(self.masks[mask_name])
                    out = out * mask
                out = block.bn2(out)
                
                if block.downsample is not None:
                    identity = block.downsample(identity)
                    shortcut_mask_name = f'{layer_name}.{i}.downsample.0'
                    if shortcut_mask_name in self.masks:
                        shortcut_mask = self._approx_sign(self.masks[shortcut_mask_name])
                        identity = identity * shortcut_mask
                
                out += identity
                out = F.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.student.fc(out)  # [B, 1]
        return out

    def train(self):
        print("\n" + "="*70)
        print("Starting PDD Training - 1-CLASS IMPLEMENTATION")
        print("="*70)
        print(f"KD Loss: MSE on logits")
        print(f"Alpha (KD weight): {self.args.alpha}")
        print(f"Learning Rate: {self.args.lr}")
        print(f"Mask Learning Rate: {self.args.lr * 5.0}")
        print(f"Total Masks: {len(self.masks)}")
        print("="*70 + "\n")
        
        for epoch in range(self.args.epochs):
            self.student.train()
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets_float = targets.float()  # ✅ for BCE
                
                self.optimizer.zero_grad()
                
                student_outputs = self._forward_with_masks(inputs).squeeze(1)  # [B]
                
                with torch.no_grad():
                    teacher_outputs = self.teacher(inputs).squeeze(1)  # [B]
                
                # Classification loss (BCE)
                ce_loss = self.criterion(student_outputs, targets_float)
                
                # KD loss: MSE between logits
                kd_loss = F.mse_loss(student_outputs, teacher_outputs)
                
                total_loss = self.args.alpha * kd_loss + (1 - self.args.alpha) * ce_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(list(self.masks.values()), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += total_loss.item()
                kd_loss_total += kd_loss.item()
                ce_loss_total += ce_loss.item()
                
                preds = (student_outputs > 0).long()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            test_acc = self.evaluate()
            self.scheduler.step()
            pruning_ratio = self._calculate_pruning_ratio()
            
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train: Loss={train_loss/len(self.train_loader):.4f}, Acc={train_acc:.2f}%")
            print(f"Test: Acc={test_acc:.2f}%")
            print(f"Losses: KD={kd_loss_total/len(self.train_loader):.4f}, CE={ce_loss_total/len(self.train_loader):.4f}")
            print(f"Pruning Ratio: {pruning_ratio:.2f}%")
            self._print_mask_stats()
            
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_masks = {name: mask.clone().detach() for name, mask in self.masks.items()}
                print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        if self.best_masks is not None:
            for name in self.masks.keys():
                self.masks[name].data = self.best_masks[name].data
        
        print(f"\nTraining Complete! Best Accuracy: {self.best_acc:.2f}%")

    def _print_mask_stats(self):
        raw_means, raw_mins, raw_maxs, approx_means, pruned_counts = [], [], [], [], []
        for mask in self.masks.values():
            raw = mask.detach()
            approx = self._approx_sign(mask).detach()
            raw_means.append(raw.mean().item())
            raw_mins.append(raw.min().item())
            raw_maxs.append(raw.max().item())
            approx_means.append(approx.mean().item())
            pruned_counts.append((raw < -1).sum().item())
        print(f" Raw Masks: Avg={np.mean(raw_means):.3f}, Min={np.min(raw_mins):.3f}, Max={np.max(raw_maxs):.3f}")
        print(f" After ApproxSign: Avg={np.mean(approx_means):.3f}")
        print(f" Channels pruned (raw<-1): {np.sum(pruned_counts)}")

    def evaluate(self):
        self.student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._forward_with_masks(inputs).squeeze(1)
                preds = (outputs > 0).long()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
        return 100. * correct / total

    def _calculate_pruning_ratio(self):
        total, pruned = 0, 0
        for mask in self.masks.values():
            raw = mask.detach()
            total += raw.numel()
            pruned += (raw < -1).sum().item()
        return 100. * pruned / total if total > 0 else 0.0

    def get_masks(self):
        binary_masks = {}
        print(f"\n🔍 Generating binary masks...")
        for name, mask in self.masks.items():
            score = self._approx_sign(mask).detach().squeeze()
            binary_mask = (score > 0.0).float()
            binary_masks[name] = binary_mask
            kept = binary_mask.sum().item()
            total = binary_mask.numel()
            print(f"{name:30s}: {int(kept):3d}/{int(total):3d} kept")
        return binary_masks
