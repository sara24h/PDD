import torch
import torch.nn as nn
import copy
from models.resnet import resnet20


def create_pruned_resnet20(original_model, masks):
    """
    Create a new structurally pruned ResNet20 model.
    Assumes model uses 'conv1', 'layerX.Y.convZ', and 'linear' naming.
    """
    # ایجاد مدل جدید
    pruned_model = resnet20(num_classes=10)
    
    # دسترسی سریع به لایه‌ها
    orig_modules = dict(original_model.named_modules())
    new_modules = dict(pruned_model.named_modules())
    
    # ذخیرهٔ تعداد کانال‌های باقی‌مانده برای هر لایه
    channel_map = {}

    # === 1. لایه اول: conv1 + bn1 ===
    mask = masks.get('conv1', None)
    if mask is not None:
        keep = (mask.cpu().squeeze() > 0.5).nonzero(as_tuple=True)[0]
        if len(keep) == 0:
            keep = torch.tensor([0])
        channel_map['conv1'] = keep
        # کپی وزن‌های هرس‌شده
        new_modules['conv1'].weight.data = orig_modules['conv1'].weight.data[keep]
        if orig_modules['conv1'].bias is not None:
            new_modules['conv1'].bias.data = orig_modules['conv1'].bias.data[keep]
        new_modules['conv1'].out_channels = len(keep)
        # BN
        new_modules['bn1'].weight.data = orig_modules['bn1'].weight.data[keep]
        new_modules['bn1'].bias.data = orig_modules['bn1'].bias.data[keep]
        new_modules['bn1'].running_mean = orig_modules['bn1'].running_mean[keep]
        new_modules['bn1'].running_var = orig_modules['bn1'].running_var[keep]
        new_modules['bn1'].num_features = len(keep)
    else:
        # بدون هرس
        new_modules['conv1'].load_state_dict(orig_modules['conv1'].state_dict())
        new_modules['bn1'].load_state_dict(orig_modules['bn1'].state_dict())
        channel_map['conv1'] = torch.arange(orig_modules['conv1'].out_channels)

    # === 2. لایه‌های باقی‌مانده: layer1, layer2, layer3 ===
    for layer_name in ['layer1', 'layer2', 'layer3']:
        for block_idx in range(3):  # ResNet20: 3 block per layer
            prefix = f"{layer_name}.{block_idx}"
            
            # --- conv1 ---
            conv1 = f"{prefix}.conv1"
            bn1 = f"{prefix}.bn1"
            # ورودی conv1 = خروجی لایه قبلی
            if block_idx == 0:
                if layer_name == 'layer1':
                    in_keep = channel_map['conv1']
                elif layer_name == 'layer2':
                    in_keep = channel_map['layer1.2.conv2']
                else:  # layer3
                    in_keep = channel_map['layer2.2.conv2']
            else:
                in_keep = channel_map[f"{layer_name}.{block_idx-1}.conv2"]

            mask = masks.get(conv1, None)
            if mask is not None:
                out_keep = (mask.cpu().squeeze() > 0.5).nonzero(as_tuple=True)[0]
                if len(out_keep) == 0:
                    out_keep = torch.tensor([0])
                channel_map[conv1] = out_keep
                # هرس هم ورودی و هم خروجی
                w = orig_modules[conv1].weight.data[out_keep][:, in_keep]
                new_modules[conv1].weight.data = w
                new_modules[conv1].in_channels = len(in_keep)
                new_modules[conv1].out_channels = len(out_keep)
                # BN
                new_modules[bn1].weight.data = orig_modules[bn1].weight.data[out_keep]
                new_modules[bn1].bias.data = orig_modules[bn1].bias.data[out_keep]
                new_modules[bn1].running_mean = orig_modules[bn1].running_mean[out_keep]
                new_modules[bn1].running_var = orig_modules[bn1].running_var[out_keep]
                new_modules[bn1].num_features = len(out_keep)
            else:
                w = orig_modules[conv1].weight.data[:, in_keep]
                new_modules[conv1].weight.data = w
                new_modules[conv1].in_channels = len(in_keep)
                new_modules[conv1].load_state_dict(orig_modules[conv1].state_dict(), strict=False)
                new_modules[bn1].load_state_dict(orig_modules[bn1].state_dict())
                channel_map[conv1] = torch.arange(orig_modules[conv1].out_channels)

            # --- conv2 ---
            conv2 = f"{prefix}.conv2"
            bn2 = f"{prefix}.bn2"
            in_keep = channel_map[conv1]  # ورودی = خروجی conv1 همین block
            mask = masks.get(conv2, None)
            if mask is not None:
                out_keep = (mask.cpu().squeeze() > 0.5).nonzero(as_tuple=True)[0]
                if len(out_keep) == 0:
                    out_keep = torch.tensor([0])
                channel_map[conv2] = out_keep
                w = orig_modules[conv2].weight.data[out_keep][:, in_keep]
                new_modules[conv2].weight.data = w
                new_modules[conv2].in_channels = len(in_keep)
                new_modules[conv2].out_channels = len(out_keep)
                # BN
                new_modules[bn2].weight.data = orig_modules[bn2].weight.data[out_keep]
                new_modules[bn2].bias.data = orig_modules[bn2].bias.data[out_keep]
                new_modules[bn2].running_mean = orig_modules[bn2].running_mean[out_keep]
                new_modules[bn2].running_var = orig_modules[bn2].running_var[out_keep]
                new_modules[bn2].num_features = len(out_keep)
            else:
                w = orig_modules[conv2].weight.data[:, in_keep]
                new_modules[conv2].weight.data = w
                new_modules[conv2].in_channels = len(in_keep)
                new_modules[conv2].load_state_dict(orig_modules[conv2].state_dict(), strict=False)
                new_modules[bn2].load_state_dict(orig_modules[bn2].state_dict())
                channel_map[conv2] = torch.arange(orig_modules[conv2].out_channels)

            # --- shortcut (اگر وجود داشت) ---
            shortcut_conv = f"{prefix}.shortcut.0"
            shortcut_bn = f"{prefix}.shortcut.1"
            if shortcut_conv in orig_modules:
                # ورودی shortcut
                if block_idx == 0:
                    if layer_name == 'layer1':
                        in_keep_short = channel_map['conv1']
                    elif layer_name == 'layer2':
                        in_keep_short = channel_map['layer1.2.conv2']
                    else:
                        in_keep_short = channel_map['layer2.2.conv2']
                else:
                    in_keep_short = channel_map[f"{layer_name}.{block_idx-1}.conv2"]

                out_keep = channel_map[conv2]
                # کپی و هرس shortcut
                w_short = orig_modules[shortcut_conv].weight.data[out_keep][:, in_keep_short]
                new_modules[shortcut_conv].weight.data = w_short
                new_modules[shortcut_conv].in_channels = len(in_keep_short)
                new_modules[shortcut_conv].out_channels = len(out_keep)
                # BN shortcut
                new_modules[shortcut_bn].weight.data = orig_modules[shortcut_bn].weight.data[out_keep]
                new_modules[shortcut_bn].bias.data = orig_modules[shortcut_bn].bias.data[out_keep]
                new_modules[shortcut_bn].running_mean = orig_modules[shortcut_bn].running_mean[out_keep]
                new_modules[shortcut_bn].running_var = orig_modules[shortcut_bn].running_var[out_keep]
                new_modules[shortcut_bn].num_features = len(out_keep)

    # === 3. لایه خطی (classifier) ===
    final_in = len(channel_map['layer3.2.conv2'])
    pruned_model.linear = nn.Linear(final_in, 10)
    pruned_model.linear.weight.data = orig_modules['linear'].weight.data[:, :final_in]
    pruned_model.linear.bias.data = orig_modules['linear'].bias.data

    return pruned_model


class ModelPruner:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks

    def prune(self):
        print("\nPruning model based on learned masks...")
        pruned_model = create_pruned_resnet20(self.model.cpu(), self.masks)
        print("✓ Pruning completed successfully")
        return pruned_model

    def get_params_count(self):
        orig = sum(p.numel() for p in self.model.parameters())
        pruned = sum(p.numel() for p in self.prune().parameters())
        return orig, pruned

    def get_flops_count(self, input_size=(1, 3, 32, 32)):
        from thop import profile
        dummy_input = torch.randn(input_size)
        orig_flops, _ = profile(self.model.cpu(), inputs=(dummy_input,), verbose=False)
        pruned_flops, _ = profile(self.prune(), inputs=(dummy_input,), verbose=False)
        return int(orig_flops), int(pruned_flops)
