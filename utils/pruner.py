import torch
import torch.nn as nn
from models.resnet import resnet20


def create_pruned_resnet20(original_model, masks, threshold=0.5):
    """
    Create pruned ResNet20 following PDD paper methodology.
    
    Key fix: Ensures identity shortcut dimension compatibility by forcing
    conv2 output channels to match the block input channels.
    """
    
    # Create new model
    pruned_model = resnet20(num_classes=10)
    orig_modules = dict(original_model.named_modules())
    new_modules = dict(pruned_model.named_modules())
    
    # Track kept channels for each layer
    channel_map = {}
    
    print("\n" + "="*70)
    print("Creating Pruned Model (Paper-Compliant)")
    print("="*70)
    
    # ========================================
    # Phase 1: Determine kept channels
    # ========================================
    print("\nPhase 1: Analyzing masks...")
    print("-"*70)
    
    # First pass: collect all masks
    temp_channel_map = {}
    for name, module in original_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        
        if name in masks:
            mask = masks[name].cpu().squeeze()
            keep_idx = (mask > threshold).nonzero(as_tuple=True)[0]
            
            # Ensure at least 1 channel
            if len(keep_idx) == 0:
                keep_idx = torch.tensor([0])
        else:
            keep_idx = torch.arange(module.out_channels)
        
        temp_channel_map[name] = keep_idx
    
    # Second pass: adjust for identity shortcuts
    for name, module in original_model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        
        keep_idx = temp_channel_map[name].clone()
        
        # Special handling for blocks with identity shortcut
        if 'conv2' in name and 'layer' in name:
            parts = name.split('.')
            layer_name = parts[0]  # e.g., 'layer1'
            block_idx = int(parts[1])  # e.g., 0, 1, 2
            
            # Determine if this block has identity shortcut
            shortcut_name = f'{layer_name}.{block_idx}.shortcut.0'
            has_projection = shortcut_name in temp_channel_map
            
            if not has_projection:
                # Identity shortcut: conv2 output must match block input
                if block_idx == 0:
                    # First block in layer: input from previous layer or conv1
                    if layer_name == 'layer1':
                        input_name = 'conv1'
                    else:
                        prev_layer = f'layer{int(layer_name[-1])-1}'
                        input_name = f'{prev_layer}.2.conv2'
                else:
                    # Later blocks: input from previous block's conv2
                    input_name = f'{layer_name}.{block_idx-1}.conv2'
                
                if input_name in temp_channel_map:
                    expected_channels = len(temp_channel_map[input_name])
                    
                    # Force conv2 to have same number of channels as input
                    if len(keep_idx) != expected_channels:
                        # Take top-k channels or pad if needed
                        if len(keep_idx) > expected_channels:
                            keep_idx = keep_idx[:expected_channels]
                        else:
                            # Repeat channels to match
                            needed = expected_channels - len(keep_idx)
                            keep_idx = torch.cat([keep_idx, keep_idx[:needed]])
                        
                        print(f"  🔧 {name:<35} | Adjusted to {len(keep_idx)} channels (identity constraint)")
        
        kept_pct = len(keep_idx) / module.out_channels * 100
        print(f"  ✓ {name:<35} | {module.out_channels:3d} → {len(keep_idx):3d} ({kept_pct:5.1f}%)")
        
        channel_map[name] = keep_idx
    
    # ========================================
    # Phase 2: Build pruned model
    # ========================================
    print("\n" + "-"*70)
    print("Phase 2: Building pruned model...")
    print("-"*70)
    
    # Initial conv1
    print("\n[Initial Conv]")
    conv1_out = channel_map['conv1']
    new_modules['conv1'].weight = nn.Parameter(
        orig_modules['conv1'].weight.data[conv1_out].clone()
    )
    new_modules['conv1'].out_channels = len(conv1_out)
    
    # BN1
    new_modules['bn1'].weight = nn.Parameter(orig_modules['bn1'].weight.data[conv1_out].clone())
    new_modules['bn1'].bias = nn.Parameter(orig_modules['bn1'].bias.data[conv1_out].clone())
    new_modules['bn1'].running_mean = orig_modules['bn1'].running_mean[conv1_out].clone()
    new_modules['bn1'].running_var = orig_modules['bn1'].running_var[conv1_out].clone()
    new_modules['bn1'].num_features = len(conv1_out)
    
    print(f"  conv1: {orig_modules['conv1'].out_channels} → {len(conv1_out)}")
    
    # Process each layer
    for layer_idx, layer_name in enumerate(['layer1', 'layer2', 'layer3']):
        print(f"\n[{layer_name.upper()}]")
        
        for block_idx in range(3):
            block_name = f"{layer_name}.{block_idx}"
            print(f"  Block {block_idx}:")
            
            # Determine input channels
            if block_idx == 0:
                if layer_idx == 0:
                    in_channels = conv1_out
                else:
                    prev_layer = ['layer1', 'layer2', 'layer3'][layer_idx - 1]
                    in_channels = channel_map[f'{prev_layer}.2.conv2']
            else:
                in_channels = channel_map[f'{layer_name}.{block_idx-1}.conv2']
            
            # Get output channels
            conv1_name = f"{block_name}.conv1"
            conv2_name = f"{block_name}.conv2"
            conv1_out_ch = channel_map[conv1_name]
            conv2_out_ch = channel_map[conv2_name]
            
            # Prune conv1
            orig_conv1 = orig_modules[conv1_name]
            new_conv1 = new_modules[conv1_name]
            new_w1 = orig_conv1.weight.data[conv1_out_ch][:, in_channels, :, :]
            new_conv1.weight = nn.Parameter(new_w1.clone())
            new_conv1.in_channels = len(in_channels)
            new_conv1.out_channels = len(conv1_out_ch)
            
            # Prune bn1
            bn1_name = f"{block_name}.bn1"
            new_modules[bn1_name].weight = nn.Parameter(orig_modules[bn1_name].weight.data[conv1_out_ch].clone())
            new_modules[bn1_name].bias = nn.Parameter(orig_modules[bn1_name].bias.data[conv1_out_ch].clone())
            new_modules[bn1_name].running_mean = orig_modules[bn1_name].running_mean[conv1_out_ch].clone()
            new_modules[bn1_name].running_var = orig_modules[bn1_name].running_var[conv1_out_ch].clone()
            new_modules[bn1_name].num_features = len(conv1_out_ch)
            
            print(f"    conv1: {orig_conv1.out_channels} → {len(conv1_out_ch)}")
            
            # Prune conv2
            orig_conv2 = orig_modules[conv2_name]
            new_conv2 = new_modules[conv2_name]
            new_w2 = orig_conv2.weight.data[conv2_out_ch][:, conv1_out_ch, :, :]
            new_conv2.weight = nn.Parameter(new_w2.clone())
            new_conv2.in_channels = len(conv1_out_ch)
            new_conv2.out_channels = len(conv2_out_ch)
            
            # Prune bn2
            bn2_name = f"{block_name}.bn2"
            new_modules[bn2_name].weight = nn.Parameter(orig_modules[bn2_name].weight.data[conv2_out_ch].clone())
            new_modules[bn2_name].bias = nn.Parameter(orig_modules[bn2_name].bias.data[conv2_out_ch].clone())
            new_modules[bn2_name].running_mean = orig_modules[bn2_name].running_mean[conv2_out_ch].clone()
            new_modules[bn2_name].running_var = orig_modules[bn2_name].running_var[conv2_out_ch].clone()
            new_modules[bn2_name].num_features = len(conv2_out_ch)
            
            print(f"    conv2: {orig_conv2.out_channels} → {len(conv2_out_ch)}")
            
            # Handle shortcut
            shortcut_name = f"{block_name}.shortcut.0"
            if shortcut_name in orig_modules:
                # Projection shortcut
                sc_out = channel_map[shortcut_name]
                orig_sc = orig_modules[shortcut_name]
                new_sc = new_modules[shortcut_name]
                
                new_w_sc = orig_sc.weight.data[sc_out][:, in_channels, :, :]
                new_sc.weight = nn.Parameter(new_w_sc.clone())
                new_sc.in_channels = len(in_channels)
                new_sc.out_channels = len(sc_out)
                
                # Prune shortcut BN
                sc_bn = f"{block_name}.shortcut.1"
                new_modules[sc_bn].weight = nn.Parameter(orig_modules[sc_bn].weight.data[sc_out].clone())
                new_modules[sc_bn].bias = nn.Parameter(orig_modules[sc_bn].bias.data[sc_out].clone())
                new_modules[sc_bn].running_mean = orig_modules[sc_bn].running_mean[sc_out].clone()
                new_modules[sc_bn].running_var = orig_modules[sc_bn].running_var[sc_out].clone()
                new_modules[sc_bn].num_features = len(sc_out)
                
                print(f"    shortcut: {orig_sc.out_channels} → {len(sc_out)} (projection)")
            else:
                print(f"    shortcut: identity")
    
    # Final linear layer
    print("\n[Final Linear]")
    final_ch = channel_map['layer3.2.conv2']
    pruned_model.linear = nn.Linear(len(final_ch), 10)
    pruned_model.linear.weight = nn.Parameter(
        orig_modules['linear'].weight.data[:, final_ch].clone()
    )
    pruned_model.linear.bias = nn.Parameter(orig_modules['linear'].bias.data.clone())
    
    print(f"  linear: {orig_modules['linear'].in_features} → {len(final_ch)}")
    
    print("\n" + "="*70)
    print("Pruning Complete!")
    print("="*70 + "\n")
    
    return pruned_model


class ModelPruner:
    """Model pruner that handles ResNet structure correctly."""
    
    def __init__(self, model, masks, threshold=0.5):
        self.model = model
        self.masks = masks
        self.threshold = threshold
        self._pruned_model = None
    
    def prune(self):
        """Prune the model based on learned masks."""
        if self._pruned_model is None:
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
                print(f"✓ Validation successful! Output: {output.shape}")
            except Exception as e:
                print(f"✗ Validation failed: {e}")
                raise
        
        return self._pruned_model
    
    def get_params_count(self):
        """Get parameter counts."""
        orig = sum(p.numel() for p in self.model.parameters())
        pruned = sum(p.numel() for p in self.prune().parameters())
        return orig, pruned
    
    def get_flops_count(self, input_size=(1, 3, 32, 32)):
        """Calculate FLOPs."""
        try:
            from thop import profile
            
            dummy = torch.randn(input_size)
            
            self.model.cpu()
            orig_flops, _ = profile(self.model, inputs=(dummy,), verbose=False)
            
            pruned_model = self.prune()
            pruned_flops, _ = profile(pruned_model, inputs=(dummy,), verbose=False)
            
            return int(orig_flops), int(pruned_flops)
        except ImportError:
            print("Warning: thop not installed. Cannot calculate FLOPs.")
            return 0, 0
