import torch
import torch.nn as nn
import numpy as np
from models.resnet import resnet20


class ModelPruner:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self._pruned_model = None
        self._original_params = None
        self._pruned_params = None
        self._original_flops = None
        self._pruned_flops = None

    def prune(self):
        """Create a pruned model based on masks"""
        # Create a new model with adjusted architecture
        self._pruned_model = resnet20(num_classes=10)
        
        # Analyze masks to determine channel counts
        channel_configs = self._analyze_masks()
        
        # Build the pruned model
        self._build_pruned_model(channel_configs)
        
        # Copy weights from original model to pruned model
        self._copy_weights()
        
        # Validate the pruned model
        self._validate_model()
        
        return self._pruned_model

    def _analyze_masks(self):
        """Analyze masks to determine channel counts for each layer"""
        channel_configs = {}
        
        # For each layer, determine which channels to keep
        for name, mask in self.masks.items():
            # Apply ApproxSign function to get binary mask
            binary_mask = self._approx_sign(mask)
            
            # Count number of channels to keep
            keep_channels = int(binary_mask.sum().item())
            total_channels = mask.shape[1]
            
            channel_configs[name] = {
                'keep_indices': torch.where(binary_mask.squeeze())[0],
                'keep_count': keep_channels,
                'total_count': total_channels
            }
        
        return channel_configs

    def _approx_sign(self, x):
        """Differentiable approximation of sign function from the paper"""
        result = torch.zeros_like(x)
        
        # x < -1
        mask1 = (x < -1).float()
        result += mask1 * 0.0
        
        # -1 <= x < 0
        mask2 = ((x >= -1) & (x < 0)).float()
        result += mask2 * ((x + 1) ** 2 / 2)
        
        # 0 <= x < 1
        mask3 = ((x >= 0) & (x < 1)).float()
        result += mask3 * (2 * x - x**2 + 1) / 2
        
        # x >= 1
        mask4 = (x >= 1).float()
        result += mask4 * 1.0
        
        return result

    def _build_pruned_model(self, channel_configs):
        """Build the pruned model with adjusted channel counts"""
        # Adjust the first convolutional layer
        conv1_keep_count = channel_configs['conv1']['keep_count']
        self._pruned_model.conv1 = nn.Conv2d(
            3, conv1_keep_count, kernel_size=3, stride=1, padding=1, bias=False
        )
        self._pruned_model.bn1 = nn.BatchNorm2d(conv1_keep_count)
        
        # Adjust each layer
        self._adjust_layer('layer1', channel_configs, 16)
        self._adjust_layer('layer2', channel_configs, 32)
        self._adjust_layer('layer3', channel_configs, 64)
        
        # Adjust the final linear layer
        last_layer_name = 'layer3.2.conv2'
        last_layer_keep_count = channel_configs[last_layer_name]['keep_count']
        self._pruned_model.linear = nn.Linear(last_layer_keep_count, 10)

    def _adjust_layer(self, layer_name, channel_configs, base_planes):
        """Adjust a specific layer (layer1, layer2, or layer3)"""
        layer = getattr(self._pruned_model, layer_name)
        blocks = []
        
        for i in range(3):  # Each layer has 3 blocks in ResNet20
            # Get the block from the original model
            original_block = getattr(self.model, layer_name)[i]
            
            # Determine input and output channels
            block_prefix = f"{layer_name}.{i}"
            
            # For the first block in each layer, input channels come from the previous layer
            if i == 0:
                if layer_name == 'layer1':
                    in_channels = channel_configs['conv1']['keep_count']
                else:
                    # Get the output channels from the last conv of the previous layer
                    prev_layer_name = 'layer1' if layer_name == 'layer2' else 'layer2'
                    prev_block_name = f"{prev_layer_name}.2.conv2"
                    in_channels = channel_configs[prev_block_name]['keep_count']
            else:
                # For subsequent blocks, input channels come from the previous block
                prev_conv_name = f"{block_prefix}.{(i-1) % 3}.conv2"
                in_channels = channel_configs[prev_conv_name]['keep_count']
            
            # Get the output channels for this block
            conv1_name = f"{block_prefix}.conv1"
            conv2_name = f"{block_prefix}.conv2"
            out_channels = channel_configs[conv2_name]['keep_count']
            
            # Determine if we need a shortcut
            stride = 2 if (i == 0 and layer_name != 'layer1') else 1
            need_shortcut = (stride != 1 or in_channels != out_channels)
            
            # Create the block
            block = self._create_block(in_channels, out_channels, stride, need_shortcut)
            blocks.append(block)
        
        # Replace the layer with the new blocks
        setattr(self._pruned_model, layer_name, nn.Sequential(*blocks))

    def _create_block(self, in_channels, out_channels, stride, need_shortcut):
        """Create a BasicBlock with the specified channels"""
        from models.resnet import BasicBlock
        block = BasicBlock(in_channels, out_channels, stride)
        
        # Adjust the conv layers
        block.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)
        block.bn1 = nn.BatchNorm2d(out_channels)
        block.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        block.bn2 = nn.BatchNorm2d(out_channels)
        
        # Adjust the shortcut if needed
        if need_shortcut:
            block.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        return block

    def _copy_weights(self):
        """Copy weights from the original model to the pruned model"""
        # Copy weights for each layer based on the keep indices
        for name, config in self._analyze_masks().items():
            if name in self.masks:
                # Get the original and pruned layers
                original_layer = self._get_layer_by_name(self.model, name)
                pruned_layer = self._get_layer_by_name(self._pruned_model, name)
                
                if isinstance(original_layer, nn.Conv2d) and isinstance(pruned_layer, nn.Conv2d):
                    # Copy the weight tensor
                    keep_indices = config['keep_indices']
                    if 'conv1' in name or name == 'conv1':
                        # For input conv layers, keep the input channels
                        if name == 'conv1':
                            pruned_layer.weight.data = original_layer.weight.data[keep_indices]
                        else:
                            pruned_layer.weight.data = original_layer.weight.data[:, keep_indices]
                    else:
                        # For other conv layers, keep both input and output channels
                        pruned_layer.weight.data = original_layer.weight.data[keep_indices][:, keep_indices]
                    
                    # Copy bias if it exists
                    if original_layer.bias is not None and pruned_layer.bias is not None:
                        pruned_layer.bias.data = original_layer.bias.data[keep_indices]
                
                elif isinstance(original_layer, nn.BatchNorm2d) and isinstance(pruned_layer, nn.BatchNorm2d):
                    # Copy batchnorm parameters
                    keep_indices = config['keep_indices']
                    pruned_layer.weight.data = original_layer.weight.data[keep_indices]
                    pruned_layer.bias.data = original_layer.bias.data[keep_indices]
                    pruned_layer.running_mean.data = original_layer.running_mean.data[keep_indices]
                    pruned_layer.running_var.data = original_layer.running_var.data[keep_indices]

    def _get_layer_by_name(self, model, name):
        """Get a layer by its name (e.g., 'layer1.0.conv1')"""
        parts = name.split('.')
        layer = model
        
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        
        return layer

    def _validate_model(self):
        """Validate the pruned model with a dummy input"""
        dummy = torch.randn(1, 3, 32, 32)
        try:
            output = self._pruned_model(dummy)
            print("✓ Validation passed")
        except Exception as e:
            print(f"✗ Validation failed: {str(e)}")
            raise

    def get_params_count(self):
        """Get the parameter count before and after pruning"""
        if self._original_params is None:
            self._original_params = sum(p.numel() for p in self.model.parameters())
        
        if self._pruned_params is None:
            self._pruned_params = sum(p.numel() for p in self._pruned_model.parameters())
        
        return self._original_params, self._pruned_params

    def get_flops_count(self):
        """Get the FLOPs count before and after pruning"""
        if self._original_flops is None:
            self._original_flops = self._count_flops(self.model)
        
        if self._pruned_flops is None:
            self._pruned_flops = self._count_flops(self._pruned_model)
        
        return self._original_flops, self._pruned_flops

    def _count_flops(self, model):
        """Count FLOPs for a model"""
        try:
            from thop import profile
            dummy = torch.randn(1, 3, 32, 32)
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            return flops
        except ImportError:
            print("Thop not installed. Using simple FLOPs counting.")
            flops = 0
            dummy = torch.randn(1, 3, 32, 32)
            
            def hook_fn(module, input, output):
                nonlocal flops
                if isinstance(module, nn.Conv2d):
                    # Calculate FLOPs for conv layer
                    batch_size, input_channels, input_height, input_width = input[0].size()
                    output_channels, output_height, output_width = output.size()
                    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * input_channels
                    flops += kernel_ops * output_height * output_width * output_channels
                elif isinstance(module, nn.Linear):
                    # Calculate FLOPs for linear layer
                    input_size = input[0].size(1)
                    output_size = output.size(1)
                    flops += input_size * output_size
            
            # Register hooks
            hooks = []
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks.append(module.register_forward_hook(hook_fn))
            
            # Forward pass
            with torch.no_grad():
                model(dummy)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return flops
