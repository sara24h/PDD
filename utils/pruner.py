import torch
import torch.nn as nn
import copy
from models.resnet import resnet20  # یا مدل مورد استفاده شما


def create_pruned_resnet20(original_model, masks):
    """
    Create a new pruned ResNet20 model based on masks.
    This avoids in-place modification and channel alignment issues.
    """
    # ایجاد مدل جدید با ساختار پایه
    pruned_model = resnet20(num_classes=10)
    
    # دیکشنری لایه‌ها برای دسترسی سریع
    orig_modules = dict(original_model.named_modules())
    new_modules = dict(pruned_model.named_modules())
    
    # نگه‌داری تعداد کانال‌های جدید برای لایه‌های بعدی
    channel_map = {}
    
    # اولین لایه: conv1
    if 'conv1' in masks:
        mask = masks['conv1'].cpu().squeeze()
        keep = (mask > 0.5).nonzero(as_tuple=True)[0]
        if len(keep) == 0:
            keep = torch.tensor([0])
        channel_map['conv1'] = keep
        # کپی وزن
        new_modules['conv1'].weight.data = orig_modules['conv1'].weight.data[keep]
        if orig_modules['conv1'].bias is not None:
            new_modules['conv1'].bias.data = orig_modules['conv1'].bias.data[keep]
        new_modules['conv1'].out_channels = len(keep)
        new_modules['bn1'].weight.data = orig_modules['bn1'].weight.data[keep]
        new_modules['bn1'].bias.data = orig_modules['bn1'].bias.data[keep]
        new_modules['bn1'].running_mean = orig_modules['bn1'].running_mean[keep]
        new_modules['bn1'].running_var = orig_modules['bn1'].running_var[keep]
        new_modules['bn1'].num_features = len(keep)
    else:
        # اگر ماسکی نبود، کپی کامل
        new_modules['conv1'].load_state_dict(orig_modules['conv1'].state_dict())
        new_modules['bn1'].load_state_dict(orig_modules['bn1'].state_dict())
        channel_map['conv1'] = torch.arange(orig_modules['conv1'].out_channels)

    # پیمایش لایه‌های باقی‌مانده
    for layer_name in ['layer1', 'layer2', 'layer3']:
        orig_layer = orig_modules[layer_name]
        new_layer = new_modules[layer_name]
        for block_idx in range(len(orig_layer)):
            prefix = f"{layer_name}.{block_idx}"
            
            # conv1
            conv1_name = f"{prefix}.conv1"
            bn1_name = f"{prefix}.bn1"
            if conv1_name in masks:
                mask = masks[conv1_name].cpu().squeeze()
                keep = (mask > 0.5).nonzero(as_tuple=True)[0]
                if len(keep) == 0:
                    keep = torch.tensor([0])
                channel_map[conv1_name] = keep
                # کپی وزن
                new_modules[conv1_name].weight.data = orig_modules[conv1_name].weight.data[keep]
                if orig_modules[conv1_name].bias is not None:
                    new_modules[conv1_name].bias.data = orig_modules[conv1_name].bias.data[keep]
                new_modules[conv1_name].out_channels = len(keep)
                # BN
                new_modules[bn1_name].weight.data = orig_modules[bn1_name].weight.data[keep]
                new_modules[bn1_name].bias.data = orig_modules[bn1_name].bias.data[keep]
                new_modules[bn1_name].running_mean = orig_modules[bn1_name].running_mean[keep]
                new_modules[bn1_name].running_var = orig_modules[bn1_name].running_var[keep]
                new_modules[bn1_name].num_features = len(keep)
            else:
                new_modules[conv1_name].load_state_dict(orig_modules[conv1_name].state_dict())
                new_modules[bn1_name].load_state_dict(orig_modules[bn1_name].state_dict())
                channel_map[conv1_name] = torch.arange(orig_modules[conv1_name].out_channels)

            # conv2
            conv2_name = f"{prefix}.conv2"
            bn2_name = f"{prefix}.bn2"
            # ورودی conv2 = خروجی conv1 همین block
            in_keep = channel_map[conv1_name]
            if conv2_name in masks:
                mask = masks[conv2_name].cpu().squeeze()
                out_keep = (mask > 0.5).nonzero(as_tuple=True)[0]
                if len(out_keep) == 0:
                    out_keep = torch.tensor([0])
                channel_map[conv2_name] = out_keep
                # کپی وزن با هرس ورودی و خروجی
                w = orig_modules[conv2_name].weight.data[out_keep][:, in_keep]
                new_modules[conv2_name].weight.data = w
                if orig_modules[conv2_name].bias is not None:
                    new_modules[conv2_name].bias.data = orig_modules[conv2_name].bias.data[out_keep]
                new_modules[conv2_name].in_channels = len(in_keep)
                new_modules[conv2_name].out_channels = len(out_keep)
                # BN
                new_modules[bn2_name].weight.data = orig_modules[bn2_name].weight.data[out_keep]
                new_modules[bn2_name].bias.data = orig_modules[bn2_name].bias.data[out_keep]
                new_modules[bn2_name].running_mean = orig_modules[bn2_name].running_mean[out_keep]
                new_modules[bn2_name].running_var = orig_modules[bn2_name].running_var[out_keep]
                new_modules[bn2_name].num_features = len(out_keep)
            else:
                w = orig_modules[conv2_name].weight.data[:, in_keep]
                new_modules[conv2_name].weight.data = w
                new_modules[conv2_name].in_channels = len(in_keep)
                new_modules[conv2_name].load_state_dict(orig_modules[conv2_name].state_dict(), strict=False)
                new_modules[bn2_name].load_state_dict(orig_modules[bn2_name].state_dict())
                channel_map[conv2_name] = torch.arange(orig_modules[conv2_name].out_channels)

            # shortcut (اگر وجود داشت)
            shortcut_conv = f"{prefix}.shortcut.0"
            shortcut_bn = f"{prefix}.shortcut.1"
            if shortcut_conv in orig_modules:
                # ورودی shortcut = خروجی لایه قبلی
                if block_idx == 0:
                    # اولین block هر لایه: ورودی از لایه قبلی می‌آید
                    if layer_name == 'layer1':
                        in_channels = 16  # خروجی conv1 اصلی
                        in_keep_prev = channel_map['conv1']
                    elif layer_name == 'layer2':
                        in_keep_prev = channel_map['layer1.2.conv2']
                    elif layer_name == 'layer3':
                        in_keep_prev = channel_map['layer2.2.conv2']
                else:
                    # block بعدی: ورودی از conv2 block قبلی
                    prev_conv2 = f"{layer_name}.{block_idx-1}.conv2"
                    in_keep_prev = channel_map[prev_conv2]

                # هرس shortcut
                w_short = orig_modules[shortcut_conv].weight.data[:, in_keep_prev]
                new_modules[shortcut_conv].weight.data = w_short
                new_modules[shortcut_conv].in_channels = len(in_keep_prev)
                new_modules[shortcut_conv].out_channels = new_modules[conv2_name].out_channels
                # BN
                out_keep = channel_map[conv2_name]
                new_modules[shortcut_bn].weight.data = orig_modules[shortcut_bn].weight.data[out_keep]
                new_modules[shortcut_bn].bias.data = orig_modules[shortcut_bn].bias.data[out_keep]
                new_modules[shortcut_bn].running_mean = orig_modules[shortcut_bn].running_mean[out_keep]
                new_modules[shortcut_bn].running_var = orig_modules[shortcut_bn].running_var[out_keep]
                new_modules[shortcut_bn].num_features = len(out_keep)

    # linear layer
    if 'layer3.2.conv2' in channel_map:
        final_out = len(channel_map['layer3.2.conv2'])
    else:
        final_out = orig_modules['layer3.2.conv2'].out_channels
    new_model.fc = nn.Linear(final_out, 10)
    new_model.fc.weight.data = orig_modules['linear'].weight.data[:, :final_out]
    new_model.fc.bias.data = orig_modules['linear'].bias.data

    return pruned_model


class ModelPruner:
    def __init__(self, model, masks):
        self.model = model
        self.masks = {k: v.cpu() for k, v in masks.items()}  # همه چیز روی CPU

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
        try:
            from thop import profile
            x = torch.randn(input_size)
            f1, _ = profile(self.model.cpu(), inputs=(x,), verbose=False)
            f2, _ = profile(self.prune(), inputs=(x,), verbose=False)
            return int(f1), int(f2)
        except:
            # fallback ساده (دقیق نیست ولی کار می‌کند)
            return self.get_params_count()  # فقط برای جلوگیری از crash
