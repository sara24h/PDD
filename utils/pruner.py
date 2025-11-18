import torch
import torch.nn as nn
from models.resnet import resnet20
from thop import profile

def create_pruned_resnet20(original_model, masks):
    """
    Create a pruned ResNet20 with proper channel handling.
    CRITICAL: For blocks without shortcut conv (identity), output must equal input channels.
    """
    pruned_model = resnet20(num_classes=10)
    orig_modules = dict(original_model.named_modules())
    new_modules = dict(pruned_model.named_modules())
    channel_map = {}
    print("\n--- Starting Pruning Process ---")
    # === 1. Prune first conv layer ===
    if 'conv1' in masks:
        keep_idx = (masks['conv1'].cpu().squeeze() > 0).nonzero(as_tuple=True)[0]  # تغییر به >0
        if len(keep_idx) == 0:
            keep_idx = torch.tensor([0])
        print(f"conv1: {orig_modules['conv1'].out_channels} -> {len(keep_idx)} channels")
    else:
        keep_idx = torch.arange(orig_modules['conv1'].out_channels)
        print(f"conv1: keeping all {len(keep_idx)} channels")
   
    channel_map['conv1'] = keep_idx
   
    # Apply pruning
    new_modules['conv1'].weight = nn.Parameter(
        orig_modules['conv1'].weight.data[keep_idx].clone()
    )
    new_modules['conv1'].out_channels = len(keep_idx)
   
    new_modules['bn1'].weight = nn.Parameter(orig_modules['bn1'].weight.data[keep_idx].clone())
    new_modules['bn1'].bias = nn.Parameter(orig_modules['bn1'].bias.data[keep_idx].clone())
    new_modules['bn1'].running_mean = orig_modules['bn1'].running_mean[keep_idx].clone()
    new_modules['bn1'].running_var = orig_modules['bn1'].running_var[keep_idx].clone()
    new_modules['bn1'].num_features = len(keep_idx)
    # === 2. Process each layer and block ===
    for layer_idx, layer_name in enumerate(['layer1', 'layer2', 'layer3']):
        print(f"\n--- Processing {layer_name} ---")
       
        for block_idx in range(3):
            prefix = f"{layer_name}.{block_idx}"
           
            # Get input channels
            if block_idx == 0:
                if layer_idx == 0:
                    in_channels_idx = channel_map['conv1']
                else:
                    prev_layer = ['layer1', 'layer2', 'layer3'][layer_idx - 1]
                    in_channels_idx = channel_map[f'{prev_layer}.2.out']
            else:
                in_channels_idx = channel_map[f'{layer_name}.{block_idx-1}.out']
           
            # Check if block has shortcut conv
            conv2_name = f"{prefix}.conv2"
            shortcut_name = f"{prefix}.shortcut.0"
            has_shortcut = shortcut_name in orig_modules
           
            # Determine output channels
            if has_shortcut:
                # Block has conv shortcut - can use mask freely
                if conv2_name in masks:
                    out_channels_idx = (masks[conv2_name].cpu().squeeze() > 0).nonzero(as_tuple=True)[0]  # تغییر به >0
                    if len(out_channels_idx) == 0:
                        out_channels_idx = torch.tensor([0])
                    print(f" {prefix}: {orig_modules[conv2_name].out_channels} -> {len(out_channels_idx)} channels (masked, has shortcut)")
                else:
                    out_channels_idx = torch.arange(orig_modules[conv2_name].out_channels)
                    print(f" {prefix}: keeping all {len(out_channels_idx)} channels (no mask, has shortcut)")
            else:
                # Block has identity shortcut - output MUST match input!
                print(f" {prefix}: identity shortcut detected, output must match input ({len(in_channels_idx)} channels)")
                out_channels_idx = in_channels_idx
           
            # Save output channels
            channel_map[f'{prefix}.out'] = out_channels_idx
           
            # --- Process conv1 ---
            conv1_name = f"{prefix}.conv1"
            bn1_name = f"{prefix}.bn1"
           
            if conv1_name in masks:
                conv1_out_idx = (masks[conv1_name].cpu().squeeze() > 0).nonzero(as_tuple=True)[0]  # تغییر به >0
                if len(conv1_out_idx) == 0:
                    conv1_out_idx = torch.tensor([0])
            else:
                conv1_out_idx = torch.arange(orig_modules[conv1_name].out_channels)
           
            # Prune conv1: [out_channels, in_channels, k, k]
            orig_w1 = orig_modules[conv1_name].weight.data
            new_w1 = orig_w1[conv1_out_idx][:, in_channels_idx, :, :]
            new_modules[conv1_name].weight = nn.Parameter(new_w1.clone())
            new_modules[conv1_name].in_channels = len(in_channels_idx)
            new_modules[conv1_name].out_channels = len(conv1_out_idx)
           
            # Prune bn1
            new_modules[bn1_name].weight = nn.Parameter(orig_modules[bn1_name].weight.data[conv1_out_idx].clone())
            new_modules[bn1_name].bias = nn.Parameter(orig_modules[bn1_name].bias.data[conv1_out_idx].clone())
            new_modules[bn1_name].running_mean = orig_modules[bn1_name].running_mean[conv1_out_idx].clone()
            new_modules[bn1_name].running_var = orig_modules[bn1_name].running_var[conv1_out_idx].clone()
            new_modules[bn1_name].num_features = len(conv1_out_idx)
           
            # --- Process conv2 (use out_channels_idx) ---
            bn2_name = f"{prefix}.bn2"
           
            # Prune conv2: [out_channels, in_channels, k, k]
            orig_w2 = orig_modules[conv2_name].weight.data
            new_w2 = orig_w2[out_channels_idx][:, conv1_out_idx, :, :]
            new_modules[conv2_name].weight = nn.Parameter(new_w2.clone())
            new_modules[conv2_name].in_channels = len(conv1_out_idx)
            new_modules[conv2_name].out_channels = len(out_channels_idx)
           
            # Prune bn2
            new_modules[bn2_name].weight = nn.Parameter(orig_modules[bn2_name].weight.data[out_channels_idx].clone())
            new_modules[bn2_name].bias = nn.Parameter(orig_modules[bn2_name].bias.data[out_channels_idx].clone())
            new_modules[bn2_name].running_mean = orig_modules[bn2_name].running_mean[out_channels_idx].clone()
            new_modules[bn2_name].running_var = orig_modules[bn2_name].running_var[out_channels_idx].clone()
            new_modules[bn2_name].num_features = len(out_channels_idx)
           
            # --- Process shortcut conv (if exists) ---
            if has_shortcut:
                sc_bn_name = f"{prefix}.shortcut.1"
               
                print(f" Pruning shortcut: in={len(in_channels_idx)}, out={len(out_channels_idx)}")
               
                # Prune shortcut conv
                orig_w_sc = orig_modules[shortcut_name].weight.data
                new_w_sc = orig_w_sc[out_channels_idx][:, in_channels_idx, :, :]
                new_modules[shortcut_name].weight = nn.Parameter(new_w_sc.clone())
                new_modules[shortcut_name].in_channels = len(in_channels_idx)
                new_modules[shortcut_name].out_channels = len(out_channels_idx)
               
                # Prune shortcut bn
                new_modules[sc_bn_name].weight = nn.Parameter(orig_modules[sc_bn_name].weight.data[out_channels_idx].clone())
                new_modules[sc_bn_name].bias = nn.Parameter(orig_modules[sc_bn_name].bias.data[out_channels_idx].clone())
                new_modules[sc_bn_name].running_mean = orig_modules[sc_bn_name].running_mean[out_channels_idx].clone()
                new_modules[sc_bn_name].running_var = orig_modules[sc_bn_name].running_var[out_channels_idx].clone()
                new_modules[sc_bn_name].num_features = len(out_channels_idx)
    # === 3. Prune final linear layer ===
    final_channels_idx = channel_map['layer3.2.out']
    print(f"\n--- Pruning linear layer ---")
    print(f" Input features: {orig_modules['linear'].in_features} -> {len(final_channels_idx)}")
   
    pruned_model.linear = nn.Linear(len(final_channels_idx), 10)
    pruned_model.linear.weight = nn.Parameter(
        orig_modules['linear'].weight.data[:, final_channels_idx].clone()
    )
    pruned_model.linear.bias = nn.Parameter(orig_modules['linear'].bias.data.clone())
    print("\n--- Pruning Complete ---\n")
    return pruned_model

class ModelPruner:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self._pruned_model = None

    def prune(self):
        """Prune the model (cached after first call)"""
        if self._pruned_model is None:
            print("\n" + "="*60)
            print("PRUNING MODEL BASED ON LEARNED MASKS")
            print("="*60)
            self._pruned_model = create_pruned_resnet20(self.model.cpu(), self.masks)
           
            # Validate the pruned model
            print("Validating pruned model...")
            dummy = torch.randn(1, 3, 32, 32)
            try:
                with torch.no_grad():
                    output = self._pruned_model(dummy)
                print(f"✓ Validation successful! Output shape: {output.shape}")
            except Exception as e:
                print(f"✗ Validation failed: {e}")
                raise
           
            print("="*60 + "\n")
       
        return self._pruned_model

    def get_params_count(self):
        """Calculate parameter counts"""
        orig = sum(p.numel() for p in self.model.parameters())
        pruned = sum(p.numel() for p in self.prune().parameters())
        return orig, pruned

    def get_flops_count(self, input_size=(1, 3, 32, 32)):
        """Calculate FLOPs"""
        dummy = torch.randn(input_size)
       
        # Original FLOPs
        self.model.cpu()
        orig_flops, _ = profile(self.model, inputs=(dummy,), verbose=False)
       
        # Pruned FLOPs
        pruned_model = self.prune()
        pruned_flops, _ = profile(pruned_model, inputs=(dummy,), verbose=False)
       
        return int(orig_flops), int(pruned_flops)
