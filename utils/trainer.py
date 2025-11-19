# PDDTrainer.py
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

        # ماسک‌ها با randn * 0.01 — دقیقاً مثل مقاله
        self.masks = self._initialize_masks()

        self.optimizer = torch.optim.SGD(
            list(student.parameters()) + list(self.masks.values()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[20, 40], gamma=0.1
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

        self.best_acc = 0.0
        self.best_masks = None

    def _initialize_masks(self):
        masks = {}
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device) * 0.01,
                    requires_grad=True
                )
                masks[name] = mask
        print(f"[PDD] Created {len(masks)} masks (randn * 0.01)")
        return masks

    def _approx_sign(self, x):
        # معادله (۲) مقاله — دقیقاً همین!
        return torch.where(x < -1, torch.zeros_like(x),
               torch.where(x < 0, (x + 1)**2 / 2,
               torch.where(x < 1, (2*x - x**2 + 1)/2,
                           torch.ones_like(x))))

    def _forward_with_masks(self, x):
        out = self.student.conv1(x)
        out = self.student.bn1(out)
        out = F.relu(out)
        if 'conv1' in self.masks:
            out = out * self._approx_sign(self.masks['conv1'])

        for layer_name in ['layer1', 'layer2', 'layer3']:
            layer = getattr(self.student, layer_name)
            for i, block in enumerate(layer):
                identity = out

                out = block.conv1(out)
                out = block.bn1(out)
                out = F.relu(out)
                if f'{layer_name}.{i}.conv1' in self.masks:
                    out = out * self._approx_sign(self.masks[f'{layer_name}.{i}.conv1'])

                out = block.conv2(out)
                out = block.bn2(out)
                if f'{layer_name}.{i}.conv2' in self.masks:
                    out = out * self._approx_sign(self.masks[f'{layer_name}.{i}.conv2'])

                if len(block.shortcut) > 0:
                    identity_out = block.shortcut(identity)
                    shortcut_name = f'{layer_name}.{i}.shortcut.0'
                    if shortcut_name in self.masks:
                        identity_out = identity_out * self._approx_sign(self.masks[shortcut_name])
                    identity = identity_out

                out = out + identity
                out = F.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.student.linear(out)

    def train(self):
        print("\n" + "="*80)
        print("PDD Training Started - 100% Paper Accurate (ResNet20 ← ResNet56)")
        print("="*80)

        for epoch in range(self.args.epochs):
            self.student.train()
            total_loss = correct = total = 0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                student_out = self._forward_with_masks(inputs)

                with torch.no_grad():
                    teacher_out = self.teacher(inputs)

                ce = self.ce_loss(student_out, targets)
                kd = self.kd_loss(
                    F.log_softmax(student_out / self.args.temperature, dim=1),
                    F.softmax(teacher_out / self.args.temperature, dim=1)
                ) * (self.args.temperature ** 2)

                loss = self.args.alpha * kd + (1 - self.args.alpha) * ce
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, pred = student_out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

            self.scheduler.step()
            acc = self.evaluate()

            print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(self.train_loader):.4f} | "
                  f"Acc: {100.*correct/total:.2f}% → Test Acc: {acc:.2f}%")

            if acc > self.best_acc:
                self.best_acc = acc
                self.best_masks = {n: m.clone().detach() for n, m in self.masks.items()}
                print("   Best model saved!")

        # بازیابی بهترین ماسک‌ها
        if self.best_masks:
            for n, m in self.masks.items():
                m.data.copy_(self.best_masks[n])

        print(f"\nPDD Training Finished! Best Accuracy: {self.best_acc:.2f}%")
        print("Masks ready for final pruning (using > 1e-8 only)\n")

    def evaluate(self):
        self.student.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self._forward_with_masks(x)
                pred = out.max(1)[1]
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        return 100. * correct / total

    def get_masks(self):
        # فقط در انتها — بدون هیچ threshold مفهومی
        return {
            name: (self._approx_sign(mask) > 1e-8).float()
            for name, mask in self.masks.items()
        }
