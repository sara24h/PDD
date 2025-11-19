# utils/pruner.py
import torch
import torch.nn as nn

class ModelPruner:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self._original_params = None
        self._pruned_params = None
        self._original_flops = None
        self._pruned_flops = None

    def _approx_sign(self, x):
        # دقیقاً معادله (۲) مقاله
        return torch.where(x < -1, torch.zeros_like(x),
               torch.where(x < 0, (x + 1)**2 / 2,
               torch.where(x < 1, 1/2 * (2*x - x**2 + 1),
                           torch.ones_like(x))))

    def prune(self):
        print("\n" + "="*90)
        print("PDD PRUNING — Using ApproxSign(mask) > 1e-8 → EXACTLY AS IN THE PAPER")
        print("="*90)

        keep_indices = {}
        for name, mask in self.masks.items():
            scores = mask.squeeze()  # (1,C,1,1) → (C)
            approx = self._approx_sign(scores)
            keep_idx = torch.where(approx > 1e-8)[0]

            total = scores.numel()
            kept = len(keep_idx)
            ratio = 100.0 * (total - kept) / total if total > 0 else 0

            keep_indices[name] = keep_idx
            print(f"{name:45s} → {kept:3d}/{total:3d} kept | {ratio:5.2f}% pruned")

        print("\nBuilding pruned model...")
        pruned_model = self._build_pruned_model(keep_indices)
        print("Copying weights...")
        self._copy_weights(pruned_model, keep_indices)
        self._calc_stats(pruned_model)

        op, pp = self.get_params_count()
        of, pf = self.get_flops_count()
        print("\n" + "="*90)
        print("FINAL RESULT — MUST BE EXACTLY LIKE TABLE 1")
        print(f"Params : {op:,} → {pp:,}   ({100*(1-pp/op):.2f}% reduction)")
        print(f"FLOPs  : {of:,} → {pf:,}   ({100*(1-pf/of):.2f}% reduction)")
        print("="*90)

        return pruned_model

    def _build_pruned_model(self, keep):
        from models.resnet import ResNet, BasicBlock
        pruned = ResNet(BasicBlock, [3, 3, 3], num_classes=10).to(next(self.model.parameters()).device)

        # conv1
        if 'conv1' in keep:
            c = len(keep['conv1'])
            pruned.conv1 = nn.Conv2d(3, c, 3, 1, 1, bias=False)
            pruned.bn1 = nn.BatchNorm2d(c)

        prev_c = len(keep['conv1']) if 'conv1' in keep else 16

        for stage_idx, stage_name in enumerate(['layer1', 'layer2', 'layer3']):
            stage = getattr(pruned, stage_name)
            for i in range(3):
                b = stage[i]
                c1_name = f'{stage_name}.{i}.conv1'
                c2_name = f'{stage_name}.{i}.conv2'

                c1 = len(keep[c1_name]) if c1_name in keep else (16 if stage_idx==0 else 32 if stage_idx==1 else 64)
                c2 = len(keep[c2_name]) if c2_name in keep else c1

                stride = 2 if stage_idx > 0 and i == 0 else 1

                b.conv1 = nn.Conv2d(prev_c, c1, 3, stride, 1, bias=False)
                b.bn1 = nn.BatchNorm2d(c1)
                b.conv2 = nn.Conv2d(c1, c2, 3, 1, 1, bias=False)
                b.bn2 = nn.BatchNorm2d(c2)

                if stride != 1 or prev_c != c2:
                    b.shortcut = nn.Sequential(
                        nn.Conv2d(prev_c, c2, 1, stride, bias=False),
                        nn.BatchNorm2d(c2)
                    )

                prev_c = c2

        pruned.linear = nn.Linear(prev_c, 10)
        return pruned

    def _copy_weights(self, pruned, keep):
        sd = self.model.state_dict()
        psd = pruned.state_dict()

        # conv1
        if 'conv1' in keep:
            idx = keep['conv1']
            psd['conv1.weight'].copy_(sd['conv1.weight'][idx])
            for n in ['weight', 'bias', 'running_mean', 'running_var']:
                if f'bn1.{n}' in sd:
                    psd[f'bn1.{n}'].copy_(sd[f'bn1.{n}'][idx])

        # layers
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
                        prev_c2 = f'{"layer1" if stage_name=="layer2" else "layer2"}.2.conv2' if i==0 else f'{stage_name}.{i-1}.conv2'
                        in_idx = keep.get(prev_c2, slice(None))
                    psd[f'{prefix}.conv1.weight'].copy_(sd[f'{prefix}.conv1.weight'][out_idx][:, in_idx])
                    for n in ['weight', 'bias', 'running_mean', 'running_var']:
                        psd[f'{prefix}.bn1.{n}'].copy_(sd[f'{prefix}.bn1.{n}'][out_idx])

                # conv2
                if c2_name in keep:
                    out_idx = keep[c2_name]
                    in_idx = keep.get(c1_name, slice(None))
                    psd[f'{prefix}.conv2.weight'].copy_(sd[f'{prefix}.conv2.weight'][out_idx][:, in_idx])
                    for n in ['weight', 'bias', 'running_mean', 'running_var']:
                        psd[f'{prefix}.bn2.{n}'].copy_(sd[f'{prefix}.bn2.{n}'][out_idx])

                # shortcut
                if 'shortcut.0.weight' in sd.keys() and len(pruned._modules[stage_name][i].shortcut) > 0:
                    if i == 0 and stage_name == 'layer1':
                        in_idx = keep.get('conv1', slice(None))
                    else:
                        prev = f'{"layer1" if stage_name=="layer2" else "layer2"}.2.conv2' if i==0 else f'{stage_name}.{i-1}.conv2'
                        in_idx = keep.get(prev, slice(None))
                    out_idx = keep.get(c2_name, slice(None))
                    psd[f'{prefix}.shortcut.0.weight'].copy_(sd[f'{prefix}.shortcut.0.weight'][out_idx][:, in_idx])
                    for n in ['weight', 'bias', 'running_mean', 'running_var']:
                        psd[f'{prefix}.shortcut.1.{n}'].copy_(sd[f'{prefix}.shortcut.1.{n}'][out_idx])

        # linear
        last = 'layer3.2.conv2'
        if last in keep:
            idx = keep[last]
            psd['linear.weight'].copy_(sd['linear.weight'][:, idx])
            psd['linear.bias'].copy_(sd['linear.bias'])

        pruned.load_state_dict(psd)

    def _calc_stats(self, pruned):
        def params(m): return sum(p.numel() for p in m.parameters())
        def flops(m):
            flops = 0
            h, w = 32, 32
            # این دقیقاً همون محاسبه مقاله است
            for name, module in m.named_modules():
                if isinstance(module, nn.Conv2d):
                    flops += module.weight.numel() * h * w
                    h = (h + 2*module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
                    w = (w + 2*module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1
                elif isinstance(module, nn.Linear):
                    flops += module.in_features * module.out_features * 2
            return flops

        self._original_params = params(self.model)
        self._pruned_params = params(pruned)
        self._original_flops = flops(self.model)
        self._pruned_flops = flops(pruned)

    def get_params_count(self):
        return self._original_params, self._pruned_params
    def get_flops_count(self):
        return self._original_flops, self._pruned_flops
