import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ApproxSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
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
        grad_mask = torch.zeros_like(x)
        mask1 = (x >= -1) & (x < 0)
        grad_mask[mask1] = x[mask1] + 1
        mask2 = (x >= 0) & (x < 1)
        grad_mask[mask2] = 1 - x[mask2]
        return grad_input * grad_mask


class PDDTrainer:
    def __init__(self, student, teacher, train_loader, test_loader, device, args):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        
        # ذخیره نام لایه‌های conv اصلی
        self.conv_names = []
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                self.conv_names.append(name)
        
        # ایجاد ماسک‌ها با نام کامل لایه
        self.masks = {}
        for name in self.conv_names:
            module = dict(self.student.named_modules())[name]
            mask = nn.Parameter(
                torch.randn(module.out_channels, device=self.device) * 0.5,
                requires_grad=True
            )
            self.masks[name] = mask
        
        self.mask_params = list(self.masks.values())
        print(f"Initialized {len(self.masks)} learnable masks")
        
        # Optimizer: تمام پارامترهای مدل + ماسک‌ها
        all_params = list(self.student.parameters()) + self.mask_params
        self.optimizer = torch.optim.SGD(
            all_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.lr_decay_epochs,
            gamma=args.lr_decay_rate
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.original_forward = {}  # ذخیره forward اصلی

        # جایگزینی موقت forward
        self._replace_forward()

    def _replace_forward(self):
        """موقتاً forward لایه‌های conv را برای اعمال ماسک تغییر بده"""
        def make_hook(name):
            def forward_hook(module, inp, out):
                binary_mask = ApproxSign.apply(self.masks[name])
                return out * binary_mask.view(1, -1, 1, 1)
            return forward_hook

        for name in self.conv_names:
            module = dict(self.student.named_modules())[name]
            self.original_forward[name] = module.register_forward_hook(
                make_hook(name)
            )

    def _restore_forward(self):
        """حذف hook‌ها و بازگردانی forward اصلی"""
        for hook in self.original_forward.values():
            hook.remove()
        self.original_forward.clear()

    def distillation_loss(self, student_logits, teacher_logits, temperature):
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kl_loss * (temperature ** 2)

    def train_epoch(self, epoch):
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
            
            student_logits = self.student(inputs)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            ce_loss = self.ce_loss(student_logits, targets)
            distill_loss = self.distillation_loss(
                student_logits, teacher_logits, self.args.temperature
            )
            loss = self.args.alpha * distill_loss + (1 - self.args.alpha) * ce_loss
            loss.backward()
            self.optimizer.step()
            
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
        
        return (total_loss / len(self.train_loader),
                distill_loss_sum / len(self.train_loader),
                ce_loss_sum / len(self.train_loader),
                100. * correct / total)

    def evaluate(self):
        self.student.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.student(inputs)
                loss = self.ce_loss(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return test_loss / len(self.test_loader), 100. * correct / total

    def train(self):
        print(f"\nStarting Pruning During Distillation...")
        print(f"Temperature: {self.args.temperature}, Alpha: {self.args.alpha}")
        best_acc = 0.0
        
        for epoch in range(self.args.epochs):
            train_loss, distill_loss, ce_loss, train_acc = self.train_epoch(epoch)
            test_loss, test_acc = self.evaluate()
            self.scheduler.step()
            
            # محاسبه pruning ratio
            total_channels = sum(m.numel() for m in self.masks.values())
            kept_channels = sum((ApproxSign.apply(m) > 0.5).sum().item() for m in self.masks.values())
            pruning_ratio = (1 - kept_channels / total_channels) * 100 if total_channels > 0 else 0
            
            print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print(f"Distill Loss: {distill_loss:.4f} | CE Loss: {ce_loss:.4f}")
            print(f"Current Pruning Ratio: {pruning_ratio:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                print(f"New best accuracy: {best_acc:.2f}%")
        
        # قبل از ذخیره، forward اصلی بازگردانده شود
        self._restore_forward()
        print(f"\nTraining completed!")
        print(f"Best Test Accuracy: {best_acc:.2f}%")
        return best_acc

    def get_masks(self):
        """بازگرداندن ماسک‌های باینری با نام کامل لایه"""
        binary_masks = {}
        with torch.no_grad():
            for name, mask in self.masks.items():
                binary_mask = ApproxSign.apply(mask)
                binary_masks[name] = (binary_mask > 0.5).float()
        return binary_masks
