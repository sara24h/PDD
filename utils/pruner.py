import torch
import torch.nn as nn
from models.resnet import resnet20
from thop import profile


def create_pruned_resnet20(original_model, masks, threshold=0.5):
    """
    Create pruned ResNet20 that matches paper methodology.
    
    Key fixes:
    1. ALL conv layers use their masks independently
    2. Identity shortcut handled via channel padding/selection
    3. No forced channel equality constraints
    """
    
    # Create new model structure (will be modified)
    pruned_model = resnet20(num_classes=10)
    orig_modules = dict(original_model.named_modules())
    new_modules = dict(pruned_model.named_modules())
    
    # Track channel indices after pruning
    channel_map = {}
    
    print("\n" + "="*70)
    print("PRUNING MODEL BASED ON LEARNED MASKS (Paper-Compliant)")
    print("="*70)
    
    # ========================================
    # Phase 1: Determine kept channels for each layer
    # ========================================
    print("\nPhase 1: Analyzing masks and determining kept channels...")
    print("-"*70)
    
    for name, module in original_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        
        if name in masks:
            # Use mask to determine kept channels
            mask = masks[name].cpu().squeeze()
            keep_idx = (mask > threshold).nonzero(as_tuple=True)[0]
            
            # Ensure at least 1 channel is kept
            if len(keep_idx) == 0:
                keep_idx = torch.tensor([0])
                print(f"  ⚠ {name:<35} | All channels pruned! Keeping channel 0")
            
            kept_ratio = len(keep_idx) / module.out_channels * 100
            print(f"  ✓ {name:<35} | {module.out_channels:3d} → {len(keep_idx):3d} channels ({kept_ratio:5.1f}% kept)")
        else:
            # No mask: keep all channels
            keep_idx = torch.arange(module.out_channels)
            print(f"  - {name:<35} | {module.out_channels:3d} → {len(keep_idx):3d} channels (no mask)")
        
        channel_map[name] = keep_idx
    
    # ========================================
    # Phase 2: Build pruned model layer by layer
    # ========================================
    print("\n" + "-"*70)
    print("Phase 2: Building pruned model...")
    print("-"*70)
    
    # --- Initial conv1 + bn1 ---
    print("\n[Initial Conv Block]")
    conv1_out = channel_map['conv1']
    
    new_modules['conv1'].weight = nn.Parameter(
        orig_modules['conv1'].weight.data[conv1_out].clone()
    )
    new_modules['conv1'].out_channels = len(conv1_out)
    
    new_modules['bn1'].weight = nn.Parameter(orig_modules['bn1'].weight.data[conv1_out].clone())
    new_modules['bn1'].bias = nn.Parameter(orig_modules['bn1'].bias.data[conv1_out].clone())
    new_modules['bn1'].running_mean = orig_modules['bn1'].running_mean[conv1_out].clone()
    new_modules['bn1'].running_var = orig_modules['bn1'].running_var[conv1_out].clone()
    new_modules['bn1'].num_features = len(conv1_out)
    
    print(f"  conv1: {orig_modules['conv1'].out_channels} → {len(conv1_out)} channels")
    
    # --- Process each layer (layer1, layer2, layer3) ---
    for layer_idx, layer_name in enumerate(['layer1', 'layer2', 'layer3']):
        print(f"\n[{layer_name.upper()}]")
        
        for block_idx in range(3):
            block_name = f"{layer_name}.{block_idx}"
            print(f"  Block {block_idx}:")
            
            # Determine input channels
            if block_idx == 0:
                if layer_idx == 0:
                    in_channels_idx = conv1_out
                else:
                    prev_layer = ['layer1', 'layer2', 'layer3'][layer_idx - 1]
                    in_channels_idx = channel_map[f'{prev_layer}.2.conv2']
            else:
                in_channels_idx = channel_map[f'{layer_name}.{block_idx-1}.conv2']
            
            # Get masks for this block
            conv1_name = f"{block_name}.conv1"
            conv2_name = f"{block_name}.conv2"
            shortcut_name = f"{block_name}.shortcut.0"
            
            conv1_out_idx = channel_map[conv1_name]
            conv2_out_idx = channel_map[conv2_name]
            
            has_shortcut = shortcut_name in orig_modules
            
            # --- Prune conv1 ---
            orig_conv1 = orig_modules[conv1_name]
            new_conv1 = new_modules[conv1_name]
            
            new_w1 = orig_conv1.weight.data[conv1_out_idx][:, in_channels_idx, :, :]
            new_conv1.weight = nn.Parameter(new_w1.clone())
            new_conv1.in_channels = len(in_channels_idx)
            new_conv1.out_channels = len(conv1_out_idx)
            
            # Prune bn1
            bn1_name = f"{block_name}.bn1"
            new_modules[bn1_name].weight = nn.Parameter(orig_modules[bn1_name].weight.data[conv1_out_idx].clone())
            new_modules[bn1_name].bias = nn.Parameter(orig_modules[bn1_name].bias.data[conv1_out_idx].clone())
            new_modules[bn1_name].running_mean = orig_modules[bn1_name].running_mean[conv1_out_idx].clone()
            new_modules[bn1_name].running_var = orig_modules[bn1_name].running_var[conv1_out_idx].clone()
            new_modules[bn1_name].num_features = len(conv1_out_idx)
            
            print(f"    conv1: {orig_conv1.out_channels} → {len(conv1_out_idx)} channels")
            
            # --- Prune conv2 ---
            orig_conv2 = orig_modules[conv2_name]
            new_conv2 = new_modules[conv2_name]
            
            new_w2 = orig_conv2.weight.data[conv2_out_idx][:, conv1_out_idx, :, :]
            new_conv2.weight = nn.Parameter(new_w2.clone())
            new_conv2.in_channels = len(conv1_out_idx)
            new_conv2.out_channels = len(conv2_out_idx)
            
            # Prune bn2
            bn2_name = f"{block_name}.bn2"
            new_modules[bn2_name].weight = nn.Parameter(orig_modules[bn2_name].weight.data[conv2_out_idx].clone())
            new_modules[bn2_name].bias = nn.Parameter(orig_modules[bn2_name].bias.data[conv2_out_idx].clone())
            new_modules[bn2_name].running_mean = orig_modules[bn2_name].running_mean[conv2_out_idx].clone()
            new_modules[bn2_name].running_var = orig_modules[bn2_name].running_var[conv2_out_idx].clone()
            new_modules[bn2_name].num_features = len(conv2_out_idx)
            
            print(f"    conv2: {orig_conv2.out_channels} → {len(conv2_out_idx)} channels")
            
            # --- Handle shortcut ---
            if has_shortcut:
                # Projection shortcut: prune it
                shortcut_out_idx = channel_map[shortcut_name]
                
                orig_shortcut = orig_modules[shortcut_name]
                new_shortcut = new_modules[shortcut_name]
                
                new_w_sc = orig_shortcut.weight.data[shortcut_out_idx][:, in_channels_idx, :, :]
                new_shortcut.weight = nn.Parameter(new_w_sc.clone())
                new_shortcut.in_channels = len(in_channels_idx)
                new_shortcut.out_channels = len(shortcut_out_idx)
                
                # Prune shortcut bn
                sc_bn_name = f"{block_name}.shortcut.1"
                new_modules[sc_bn_name].weight = nn.Parameter(orig_modules[sc_bn_name].weight.data[shortcut_out_idx].clone())
                new_modules[sc_bn_name].bias = nn.Parameter(orig_modules[sc_bn_name].bias.data[shortcut_out_idx].clone())
                new_modules[sc_bn_name].running_mean = orig_modules[sc_bn_name].running_mean[shortcut_out_idx].clone()
                new_modules[sc_bn_name].running_var = orig_modules[sc_bn_name].running_var[shortcut_out_idx].clone()
                new_modules[sc_bn_name].num_features = len(shortcut_out_idx)
                
                print(f"    shortcut: {orig_shortcut.out_channels} → {len(shortcut_out_idx)} channels (projection)")
                
                # Verify dimensions match
                if len(conv2_out_idx) != len(shortcut_out_idx):
                    print(f"    ⚠ WARNING: conv2 out ({len(conv2_out_idx)}) != shortcut out ({len(shortcut_out_idx)})")
            else:
                # Identity shortcut: dimensions must match
                print(f"    shortcut: identity (channels must match)")
                
                if len(conv2_out_idx) != len(in_channels_idx):
                    print(f"    ⚠ CRITICAL: Identity shortcut dimension mismatch!")
                    print(f"       conv2 out: {len(conv2_out_idx)}, input: {len(in_channels_idx)}")
                    print(f"    → This should not happen with proper mask constraints!")
                    
                    # Emergency fix: force conv2 to match input dimensions
                    # This is a fallback - ideally masks should prevent this
                    print(f"    → Applying emergency channel selection...")
                    if len(conv2_out_idx) > len(in_channels_idx):
                        conv2_out_idx = conv2_out_idx[:len(in_channels_idx)]
                    else:
                        # Pad with duplicates
                        padding = len(in_channels_idx) - len(conv2_out_idx)
                        conv2_out_idx = torch.cat([conv2_out_idx, conv2_out_idx[:padding]])
                    
                    # Re-prune conv2 with fixed indices
                    new_w2 = orig_conv2.weight.data[conv2_out_idx][:, conv1_out_idx, :, :]
                    new_conv2.weight = nn.Parameter(new_w2.clone())
                    new_conv2.out_channels = len(conv2_out_idx)
                    
                    # Update channel_map
                    channel_map[conv2_name] = conv2_out_idx
    
    # --- Final linear layer ---
    print("\n[Final Linear Layer]")
    final_channels = channel_map['layer3.2.conv2']
    
    pruned_model.linear = nn.Linear(len(final_channels), 10)
    pruned_model.linear.weight = nn.Parameter(
        orig_modules['linear'].weight.data[:, final_channels].clone()
    )
    pruned_model.linear.bias = nn.Parameter(orig_modules['linear'].bias.data.clone())
    
    print(f"  linear: {orig_modules['linear'].in_features} → {len(final_channels)} input features")
    
    print("\n" + "="*70)
    print("PRUNING COMPLETE")
    print("="*70 + "\n")
    
    return pruned_model


class ModelPruner:
    """
    Model pruner that correctly handles ResNet structure.
    """
    
    def __init__(self, model, masks, threshold=0.5):
        self.model = model
        self.masks = masks
        self.threshold = threshold
        self._pruned_model = None
    
    def prune(self):
        """Prune the model based on learned masks."""
        if self._pruned_model is None:
            print("\n" + "="*70)
            print("STARTING MODEL PRUNING")
            print("="*70)
            
            self._pruned_model = create_pruned_resnet20(
                self.model.cpu(), 
                self.masks, 
                self.threshold
            )
            
            # Validate
            print("Validating pruned model...")
            dummy = torch.randn(1, 3, 32, 32)
            try:
                with torch.no_grad():
                    output = self._pruned_model(dummy)
                print(f"✓ Validation successful! Output shape: {output.shape}")
            except Exception as e:
                print(f"✗ Validation failed: {e}")
                raise
            
            print("="*70 + "\n")
        
        return self._pruned_model
    
    def get_params_count(self):
        """Calculate parameter counts."""
        orig = sum(p.numel() for p in self.model.parameters())
        pruned = sum(p.numel() for p in self.prune().parameters())
        return orig, pruned
    
    def get_flops_count(self, input_size=(1, 3, 32, 32)):
        """Calculate FLOPs using thop."""
        dummy = torch.randn(input_size)
        
        # Original FLOPs
        self.model.cpu()
        orig_flops, _ = profile(self.model, inputs=(dummy,), verbose=False)
        
        # Pruned FLOPs
        pruned_model = self.prune()
        pruned_flops, _ = profile(pruned_model, inputs=(dummy,), verbose=False)
        
        return int(orig_flops), int(pruned_flops)
