import torch
import torch.nn as nn
from models.resnet import resnet20
from thop import profile


def create_pruned_resnet20(original_model, masks):
    """
    Create a new pruned ResNet20 with consistent channel dimensions.
    Enforces that conv2 and shortcut in each block have the same output channels.
    """
    pruned_model = resnet20(num_classes=10)
    orig_modules = dict(original_model.named_modules())
    new_modules = dict(pruned_model.named_modules())
    channel_map = {}

    # === 1. First conv layer ===
    if 'conv1' in masks:
        keep = (masks['conv1'].cpu().squeeze() > 0.5).nonzero(as_tuple=True)[0]
        if len(keep) == 0:
            keep = torch.tensor([0])
        channel_map['conv1'] = keep
        new_modules['conv1'].weight.data = orig_modules['conv1'].weight.data[keep]
        if orig_modules['conv1'].bias is not None:
            new_modules['conv1'].bias.data = orig_modules['conv1'].bias.data[keep]
        new_modules['conv1'].out_channels = len(keep)
        # BatchNorm1
        new_modules['bn1'].weight.data = orig_modules['bn1'].weight.data[keep]
        new_modules['bn1'].bias.data = orig_modules['bn1'].bias.data[keep]
        new_modules['bn1'].running_mean = orig_modules['bn1'].running_mean[keep]
        new_modules['bn1'].running_var = orig_modules['bn1'].running_var[keep]
        new_modules['bn1'].num_features = len(keep)
    else:
        new_modules['conv1'].load_state_dict(orig_modules['conv1'].state_dict())
        new_modules['bn1'].load_state_dict(orig_modules['bn1'].state_dict())
        channel_map['conv1'] = torch.arange(orig_modules['conv1'].out_channels)

    # === 2. Main layers: layer1, layer2, layer3 ===
    for layer_name in ['layer1', 'layer2', 'layer3']:
        for block_idx in range(3):
            prefix = f"{layer_name}.{block_idx}"
            conv1_name = f"{prefix}.conv1"
            bn1_name = f"{prefix}.bn1"
            conv2_name = f"{prefix}.conv2"
            bn2_name = f"{prefix}.bn2"

            # Determine input channels for conv1
            if block_idx == 0:
                if layer_name == 'layer1':
                    in_keep = channel_map['conv1']
                elif layer_name == 'layer2':
                    in_keep = channel_map['layer1.2.conv2']
                else:  # layer3
                    in_keep = channel_map['layer2.2.conv2']
            else:
                in_keep = channel_map[f"{layer_name}.{block_idx-1}.conv2"]

            # --- Prune conv1 ---
            if conv1_name in masks:
                out_keep1 = (masks[conv1_name].cpu().squeeze() > 0.5).nonzero(as_tuple=True)[0]
                if len(out_keep1) == 0:
                    out_keep1 = torch.tensor([0])
                w1 = orig_modules[conv1_name].weight.data[out_keep1][:, in_keep]
                new_modules[conv1_name].weight.data = w1
                new_modules[conv1_name].in_channels = len(in_keep)
                new_modules[conv1_name].out_channels = len(out_keep1)
                # BN1
                new_modules[bn1_name].weight.data = orig_modules[bn1_name].weight.data[out_keep1]
                new_modules[bn1_name].bias.data = orig_modules[bn1_name].bias.data[out_keep1]
                new_modules[bn1_name].running_mean = orig_modules[bn1_name].running_mean[out_keep1]
                new_modules[bn1_name].running_var = orig_modules[bn1_name].running_var[out_keep1]
                new_modules[bn1_name].num_features = len(out_keep1)
                channel_map[conv1_name] = out_keep1
            else:
                w1 = orig_modules[conv1_name].weight.data[:, in_keep]
                new_modules[conv1_name].weight.data = w1
                new_modules[conv1_name].in_channels = len(in_keep)
                new_modules[conv1_name].load_state_dict(orig_modules[conv1_name].state_dict(), strict=False)
                new_modules[bn1_name].load_state_dict(orig_modules[bn1_name].state_dict())
                channel_map[conv1_name] = torch.arange(orig_modules[conv1_name].out_channels)

            # --- Prune conv2 (this determines block output) ---
            in_keep2 = channel_map[conv1_name]
            if conv2_name in masks:
                out_keep2 = (masks[conv2_name].cpu().squeeze() > 0.5).nonzero(as_tuple=True)[0]
                if len(out_keep2) == 0:
                    out_keep2 = torch.tensor([0])
                w2 = orig_modules[conv2_name].weight.data[out_keep2][:, in_keep2]
                new_modules[conv2_name].weight.data = w2
                new_modules[conv2_name].in_channels = len(in_keep2)
                new_modules[conv2_name].out_channels = len(out_keep2)
                # BN2
                new_modules[bn2_name].weight.data = orig_modules[bn2_name].weight.data[out_keep2]
                new_modules[bn2_name].bias.data = orig_modules[bn2_name].bias.data[out_keep2]
                new_modules[bn2_name].running_mean = orig_modules[bn2_name].running_mean[out_keep2]
                new_modules[bn2_name].running_var = orig_modules[bn2_name].running_var[out_keep2]
                new_modules[bn2_name].num_features = len(out_keep2)
                channel_map[conv2_name] = out_keep2
            else:
                w2 = orig_modules[conv2_name].weight.data[:, in_keep2]
                new_modules[conv2_name].weight.data = w2
                new_modules[conv2_name].in_channels = len(in_keep2)
                new_modules[conv2_name].load_state_dict(orig_modules[conv2_name].state_dict(), strict=False)
                new_modules[bn2_name].load_state_dict(orig_modules[bn2_name].state_dict())
                channel_map[conv2_name] = torch.arange(orig_modules[conv2_name].out_channels)

            # --- Handle shortcut (use SAME out_keep2 for output channels) ---
            shortcut_conv = f"{prefix}.shortcut.0"
            shortcut_bn = f"{prefix}.shortcut.1"
            if shortcut_conv in orig_modules:
                # Input channels for shortcut
                if block_idx == 0:
                    if layer_name == 'layer1':
                        in_keep_sc = channel_map['conv1']
                    elif layer_name == 'layer2':
                        in_keep_sc = channel_map['layer1.2.conv2']
                    else:
                        in_keep_sc = channel_map['layer2.2.conv2']
                else:
                    in_keep_sc = channel_map[f"{layer_name}.{block_idx-1}.conv2"]

                out_keep_sc = channel_map[conv2_name]  # ← MUST MATCH conv2!

                # Prune shortcut conv
                w_sc = orig_modules[shortcut_conv].weight.data[out_keep_sc][:, in_keep_sc]
                new_modules[shortcut_conv].weight.data = w_sc
                new_modules[shortcut_conv].in_channels = len(in_keep_sc)
                new_modules[shortcut_conv].out_channels = len(out_keep_sc)

                # Prune shortcut BN
                new_modules[shortcut_bn].weight.data = orig_modules[shortcut_bn].weight.data[out_keep_sc]
                new_modules[shortcut_bn].bias.data = orig_modules[shortcut_bn].bias.data[out_keep_sc]
                new_modules[shortcut_bn].running_mean = orig_modules[shortcut_bn].running_mean[out_keep_sc]
                new_modules[shortcut_bn].running_var = orig_modules[shortcut_bn].running_var[out_keep_sc]
                new_modules[shortcut_bn].num_features = len(out_keep_sc)

    # === 3. Classifier ===
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
        dummy = torch.randn(input_size)
        orig_flops, _ = profile(self.model.cpu(), inputs=(dummy,), verbose=False)
        pruned_flops, _ = profile(self.prune(), inputs=(dummy,), verbose=False)
        return int(orig_flops), int(pruned_flops)
