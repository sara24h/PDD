import torch
import torch.nn as nn
from thop import profile


class ModelPruner:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks

    def _approx_sign(self, x):
        """
        معادله (2) مقاله PDD
        """
        return torch.where(x <= -1, torch.zeros_like(x),
               torch.where(x <= 0, 0.5 * (x + 1)**2,
               torch.where(x <= 1, 0.5 * (2*x - x**2 + 1),
                           torch.ones_like(x))))

    def prune(self):
        """
        هرس کردن مدل بر اساس ماسک‌ها
        """
        print("\n" + "="*90)
        print("PDD FINAL PRUNING — USING APPROXSIGN > 1e-8")
        print("="*90)

        keep_indices = {}

        for name, mask in self.masks.items():
            scores = mask.squeeze()
            if scores.dim() > 1:
                scores = scores.flatten()

            # اعمال ApproxSign و انتخاب کانال‌های مهم
            approx = self._approx_sign(scores)
            keep_idx = torch.where(approx > 1e-8)[0]

            total = scores.numel()
            kept = len(keep_idx)
            pruned_ratio = 100.0 * (total - kept) / total

            keep_indices[name] = keep_idx
            print(f"{name:45s} → {kept:3d}/{total:3d} kept | {pruned_ratio:5.2f}% pruned")

        print("\nCreating pruned model...")
        pruned_model = self._create_pruned_model(keep_indices)
        self._copy_weights(pruned_model, keep_indices)

        return pruned_model

    def _create_pruned_model(self, keep):
        """
        ساخت مدل هرس‌شده با کانال‌های انتخاب‌شده
        """
        from models.resnet import ResNet, BasicBlock
        
        device = next(self.model.parameters()).device
        pruned = ResNet(BasicBlock, [3, 3, 3], num_classes=10).to(device)

        # Conv1
        if 'conv1' in keep:
            c = len(keep['conv1'])
            pruned.conv1 = nn.Conv2d(3, c, 3, 1, 1, bias=False).to(device)
            pruned.bn1 = nn.BatchNorm2d(c).to(device)

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

                block.conv1 = nn.Conv2d(prev_c, c1, 3, stride, 1, bias=False).to(device)
                block.bn1 = nn.BatchNorm2d(c1).to(device)
                block.conv2 = nn.Conv2d(c1, c2, 3, 1, 1, bias=False).to(device)
                block.bn2 = nn.BatchNorm2d(c2).to(device)

                if stride != 1 or prev_c != c2:
                    block.shortcut = nn.Sequential(
                        nn.Conv2d(prev_c, c2, 1, stride, bias=False).to(device),
                        nn.BatchNorm2d(c2).to(device)
                    )

                prev_c = c2

        pruned.linear = nn.Linear(prev_c, 10).to(device)
        return pruned

    def _copy_weights(self, pruned, keep):
        """
        کپی کردن وزن‌ها از مدل اصلی به مدل هرس‌شده
        """
        src = self.model.state_dict()
        dst = pruned.state_dict()

        # Conv1
        if 'conv1' in keep:
            idx = keep['conv1']
            dst['conv1.weight'].copy_(src['conv1.weight'][idx])
            for key in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                if f'bn1.{key}' in src:
                    if key == 'num_batches_tracked':
                        dst[f'bn1.{key}'].copy_(src[f'bn1.{key}'])
                    else:
                        dst[f'bn1.{key}'].copy_(src[f'bn1.{key}'][idx])

        # Layers
        for stage_name in ['layer1', 'layer2', 'layer3']:
            for i in range(3):
                prefix = f'{stage_name}.{i}'
                c1_name = f'{prefix}.conv1'
                c2_name = f'{prefix}.conv2'

                # محاسبه in_idx برای conv1
                if i == 0 and stage_name == 'layer1':
                    in_idx = keep.get('conv1', None)
                elif i == 0:
                    prev_stage = 'layer1' if stage_name == 'layer2' else 'layer2'
                    in_idx = keep.get(f'{prev_stage}.2.conv2', None)
                else:
                    in_idx = keep.get(f'{stage_name}.{i-1}.conv2', None)

                # Conv1
                if c1_name in keep:
                    out_idx = keep[c1_name]
                    if in_idx is not None:
                        dst[f'{prefix}.conv1.weight'].copy_(
                            src[f'{prefix}.conv1.weight'][out_idx][:, in_idx]
                        )
                    else:
                        dst[f'{prefix}.conv1.weight'].copy_(
                            src[f'{prefix}.conv1.weight'][out_idx]
                        )
                    
                    for key in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                        if f'{prefix}.bn1.{key}' in src:
                            if key == 'num_batches_tracked':
                                dst[f'{prefix}.bn1.{key}'].copy_(src[f'{prefix}.bn1.{key}'])
                            else:
                                dst[f'{prefix}.bn1.{key}'].copy_(src[f'{prefix}.bn1.{key}'][out_idx])

                # Conv2
                if c2_name in keep:
                    out_idx = keep[c2_name]
                    conv1_idx = keep.get(c1_name, None)
                    
                    if conv1_idx is not None:
                        dst[f'{prefix}.conv2.weight'].copy_(
                            src[f'{prefix}.conv2.weight'][out_idx][:, conv1_idx]
                        )
                    else:
                        dst[f'{prefix}.conv2.weight'].copy_(
                            src[f'{prefix}.conv2.weight'][out_idx]
                        )
                    
                    for key in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                        if f'{prefix}.bn2.{key}' in src:
                            if key == 'num_batches_tracked':
                                dst[f'{prefix}.bn2.{key}'].copy_(src[f'{prefix}.bn2.{key}'])
                            else:
                                dst[f'{prefix}.bn2.{key}'].copy_(src[f'{prefix}.bn2.{key}'][out_idx])

                    # Shortcut
                    if f'{prefix}.shortcut.0.weight' in src:
                        if in_idx is not None:
                            dst[f'{prefix}.shortcut.0.weight'].copy_(
                                src[f'{prefix}.shortcut.0.weight'][out_idx][:, in_idx]
                            )
                        else:
                            dst[f'{prefix}.shortcut.0.weight'].copy_(
                                src[f'{prefix}.shortcut.0.weight'][out_idx]
                            )
                        
                        for key in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                            if f'{prefix}.shortcut.1.{key}' in src:
                                if key == 'num_batches_tracked':
                                    dst[f'{prefix}.shortcut.1.{key}'].copy_(src[f'{prefix}.shortcut.1.{key}'])
                                else:
                                    dst[f'{prefix}.shortcut.1.{key}'].copy_(src[f'{prefix}.shortcut.1.{key}'][out_idx])

        # Linear
        last = 'layer3.2.conv2'
        if last in keep:
            idx = keep[last]
            dst['linear.weight'].copy_(src['linear.weight'][:, idx])
            dst['linear.bias'].copy_(src['linear.bias'])

        pruned.load_state_dict(dst)
