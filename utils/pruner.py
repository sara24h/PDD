# utils/pruner.py
import torch
import torch.nn as nn
from thop import profile

class ModelPruner:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks

    def _approx_sign(self, x):
        # دقیقاً معادله (۲) مقاله PDD
        return torch.where(x <= -1, torch.zeros_like(x),
               torch.where(x <= 0, 0.5 * (x + 1)**2,
               torch.where(x <= 1, 0.5 * (2*x - x**2 + 1),
                           torch.ones_like(x))))

    def prune(self):
        print("\n" + "="*90)
        print("PDD FINAL PRUNING — USING APPROXSIGN > 1e-8 (EXACTLY AS IN PAPER)")
        print("="*90)

        keep_indices = {}

        for name, mask in self.masks.items():
            scores = mask.squeeze()
            if scores.dim() > 1:
                scores = scores.flatten()

            # این دقیقاً همون چیزیه که تو مقاله نوشته شده
            approx = self._approx_sign(scores)
            keep_idx = torch.where(approx > 1e-8)[0]

            total = scores.numel()
            kept = len(keep_idx)
            pruned_ratio = 100.0 * (total - kept) / total

            keep_indices[name] = keep_idx
            print(f"{name:45s} → {kept:3d}/{total:3d} kept | {pruned_ratio:5.2f}% pruned")

        print("\nساخت مدل هرس‌شده...")
        pruned_model = self._create_pruned_model(keep_indices)
        self._copy_weights(pruned_model, keep_indices)

        # محاسبه دقیق FLOPs و Params با thop
        input_tensor = torch.randn(1, 3, 32, 32).cuda()
        original_flops, original_params = profile(self.model, inputs=(input_tensor,), verbose=False)
        pruned_flops, pruned_params = profile(pruned_model, inputs=(input_tensor,), verbose=False)

        params_reduction = 100 * (1 - pruned_params / original_params)
        flops_reduction = 100 * (1 - pruned_flops / original_flops)

        print("\n" + "="*90)
        print("نتیجه نهایی — دقیقاً مثل جدول ۱ مقاله")
        print(f"Params : {int(original_params):,} → {int(pruned_params):,}   ({params_reduction:.2f}% ↓)")
        print(f"FLOPs  : {int(original_flops):,} → {int(pruned_flops):,}   ({flops_reduction:.2f}% ↓)")
        print("="*90)

        return pruned_model

    def _create_pruned_model(self, keep):
        from models.resnet import ResNet, BasicBlock
        pruned = ResNet(BasicBlock, [3, 3, 3], num_classes=10).cuda()

        # conv1
        if 'conv1' in keep:
            c = len(keep['conv1'])
            pruned.conv1 = nn.Conv2d(3, c, 3, 1, 1, bias=False).cuda()
            pruned.bn1 = nn.BatchNorm2d(c).cuda()

        prev_c = len(keep['conv1']) if 'conv1' in keep else 16

        for stage_name in ['layer1', 'layer2', 'layer3']:
            stage = getattr(pruned, stage_name)
            for i in range(3):
                block = stage[i]
                c1_name = f'{stage_name}.{i}.conv1'
                c2_name = f'{stage_name}.{i}.conv2'

                c1 = len(keep[c1_name]) if c1_name in keep else prev_c
                c2 = len(keep[c2_name]) if c2_name in keep else c1

                stride = 2 if (stage_name != 'layer1' and i == 0) else 1

                block.conv1 = nn.Conv2d(prev_c, c1, 3, stride, 1, bias=False).cuda()
                block.bn1 = nn.BatchNorm2d(c1).cuda()
                block.conv2 = nn.Conv2d(c1, c2, 3, 1, 1, bias=False).cuda()
                block.bn2 = nn.BatchNorm2d(c2).cuda()

                if stride != 1 or prev_c != c2:
                    block.shortcut = nn.Sequential(
                        nn.Conv2d(prev_c, c2, 1, stride, bias=False).cuda(),
                        nn.BatchNorm2d(c2).cuda()
                    )

                prev_c = c2

        pruned.linear = nn.Linear(prev_c, 10).cuda()
        return pruned

    def _copy_weights(self, pruned, keep):
        src = self.model.state_dict()

        # conv1
        if 'conv1' in keep:
            idx = keep['conv1']
            pruned.conv1.weight.data = src['conv1.weight'][idx]
            for key in ['weight', 'bias', 'running_mean', 'running_var']:
                if f'bn1.{key}' in src:
                    pruned.bn1.__getattr__(key).data = src[f'bn1.{key}'][idx]

        # بقیه لایه‌ها
        for stage_name in ['layer1', 'layer2', 'layer3']:
            for i in range(3):
                prefix = f'{stage_name}.{i}'
                c1_name = f'{prefix}.conv1'
                c2_name = f'{prefix}.conv2'

                # conv1
                if c1_name in keep:
                    out_idx = keep[c1_name]
                    if i == 0 and stage_name == 'layer1':
                        in_idx = keep.get('conv1', slice(None))
                    else:
                        prev = f'layer1.2.conv2' if stage_name == 'layer2' and i == 0 else \
                               f'layer2.2.conv2' if stage_name == 'layer3' and i == 0 else \
                               f'{stage_name}.{i-1}.conv2'
                        in_idx = keep.get(prev, slice(None))

                    pruned.state_dict()[f'{prefix}.conv1.weight'].copy_(
                        src[f'{prefix}.conv1.weight'][out_idx][:, in_idx]
                    )
                    for key in ['weight', 'bias', 'running_mean', 'running_var']:
                        if f'{prefix}.bn1.{key}' in src:
                            pruned.state_dict()[f'{prefix}.bn1.{key}'].copy_(src[f'{prefix}.bn1.{key}'][out_idx])

                # conv2 + shortcut + linear
                if c2_name in keep:
                    out_idx = keep[c2_name]
                    in_idx = keep.get(c1_name, slice(None))
                    pruned.state_dict()[f'{prefix}.conv2.weight'].copy_(
                        src[f'{prefix}.conv2.weight'][out_idx][:, in_idx]
                    )
                    for key in ['weight', 'bias', 'running_mean', 'running_var']:
                        if f'{prefix}.bn2.{key}' in src:
                            pruned.state_dict()[f'{prefix}.bn2.{key}'].copy_(src[f'{prefix}.bn2.{key}'][out_idx])

                    if f'{prefix}.shortcut.0.weight' in src:
                        pruned.state_dict()[f'{prefix}.shortcut.0.weight'].copy_(
                            src[f'{prefix}.shortcut.0.weight'][out_idx][:, in_idx]
                        )
                        for key in ['weight', 'bias', 'running_mean', 'running_var']:
                            if f'{prefix}.shortcut.1.{key}' in src:
                                pruned.state_dict()[f'{prefix}.shortcut.1.{key}'].copy_(src[f'{prefix}.shortcut.1.{key}'][out_idx])

        # linear
        last = 'layer3.2.conv2'
        if last in keep:
            idx = keep[last]
            pruned.linear.weight.data = src['linear.weight'][:, idx]
            pruned.linear.bias.data = src['linear.bias']

        pruned.load_state_dict(pruned.state_dict())
