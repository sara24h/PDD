import torch
import torch.nn as nn
import copy


class ModelPruner:
    """
    Prune channels from the model based on learned masks
    """
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self.pruned_model = None
        
    def _get_kept_channels(self, mask):
        """Get indices of channels to keep based on mask"""
        # Mask shape: (1, out_channels, 1, 1)
        mask_1d = mask.squeeze()
        if mask_1d.dim() == 0:
            mask_1d = mask_1d.unsqueeze(0)
        kept_indices = torch.where(mask_1d > 0.5)[0]
        return kept_indices
    
    def _prune_conv_layer(self, conv, mask, prev_mask=None):
        """
        Prune a convolutional layer
        - mask: determines which output channels to keep
        - prev_mask: determines which input channels to keep (from previous layer)
        """
        kept_out = self._get_kept_channels(mask)
        
        if len(kept_out) == 0:
            # Keep at least one channel
            kept_out = torch.tensor([0], device=mask.device)
        
        # Create new conv layer
        new_out_channels = len(kept_out)
        
        if prev_mask is not None:
            kept_in = self._get_kept_channels(prev_mask)
            if len(kept_in) == 0:
                kept_in = torch.tensor([0], device=prev_mask.device)
            new_in_channels = len(kept_in)
        else:
            kept_in = None
            new_in_channels = conv.in_channels
        
        new_conv = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=new_out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1 if conv.groups == 1 else new_out_channels,  # Handle depthwise conv
            bias=conv.bias is not None
        )
        
        # Copy weights
        with torch.no_grad():
            if kept_in is not None:
                # Prune both input and output channels
                new_conv.weight.data = conv.weight.data[kept_out][:, kept_in].clone()
            else:
                # Only prune output channels
                new_conv.weight.data = conv.weight.data[kept_out].clone()
            
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[kept_out].clone()
        
        return new_conv, kept_out
    
    def _prune_batchnorm(self, bn, kept_channels):
        """Prune batch normalization layer"""
        if len(kept_channels) == 0:
            kept_channels = torch.tensor([0], device=bn.weight.device)
        
        new_bn = nn.BatchNorm2d(len(kept_channels))
        
        with torch.no_grad():
            new_bn.weight.data = bn.weight.data[kept_channels].clone()
            new_bn.bias.data = bn.bias.data[kept_channels].clone()
            new_bn.running_mean.data = bn.running_mean.data[kept_channels].clone()
            new_bn.running_var.data = bn.running_var.data[kept_channels].clone()
            new_bn.num_batches_tracked = bn.num_batches_tracked
        
        return new_bn
    
    def _prune_linear(self, linear, prev_mask):
        """Prune linear layer based on previous conv layer's mask"""
        if prev_mask is not None:
            kept_in = self._get_kept_channels(prev_mask)
            if len(kept_in) == 0:
                kept_in = torch.tensor([0], device=prev_mask.device)
            new_in_features = len(kept_in)
        else:
            kept_in = None
            new_in_features = linear.in_features
        
        new_linear = nn.Linear(new_in_features, linear.out_features, bias=linear.bias is not None)
        
        with torch.no_grad():
            if kept_in is not None:
                new_linear.weight.data = linear.weight.data[:, kept_in].clone()
            else:
                new_linear.weight.data = linear.weight.data.clone()
            
            if linear.bias is not None:
                new_linear.bias.data = linear.bias.data.clone()
        
        return new_linear
    
    def prune(self):
        """
        Prune the model based on masks
        Returns pruned model
        """
        print("Pruning model based on learned masks...")
        
        # Create a deep copy of the model
        pruned_model = copy.deepcopy(self.model)
        
        # Unwrap MaskedConv2d layers back to regular Conv2d
        def unwrap_module(module):
            for name, child in list(module.named_children()):
                if hasattr(child, 'conv') and hasattr(child, 'mask'):
                    # This is a MaskedConv2d, replace with its underlying conv
                    setattr(module, name, child.conv)
                else:
                    unwrap_module(child)
        
        unwrap_module(pruned_model)
        
        # Track previous layer's output mask for pruning input channels
        prev_mask = None
        layer_masks = {}
        
        # Collect masks in order
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.masks:
                layer_masks[name] = self.masks[name]
        
        # Prune each layer
        def prune_module(module, prefix=''):
            nonlocal prev_mask
            
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Conv2d) and full_name in self.masks:
                    # Prune conv layer
                    mask = self.masks[full_name]
                    new_conv, kept_out = self._prune_conv_layer(child, mask, prev_mask)
                    setattr(module, name, new_conv)
                    prev_mask = mask
                    
                elif isinstance(child, nn.BatchNorm2d) and prev_mask is not None:
                    # Prune corresponding batchnorm
                    kept_channels = self._get_kept_channels(prev_mask)
                    new_bn = self._prune_batchnorm(child, kept_channels)
                    setattr(module, name, new_bn)
                    
                elif isinstance(child, nn.Linear):
                    # Prune linear layer
                    new_linear = self._prune_linear(child, prev_mask)
                    setattr(module, name, new_linear)
                    prev_mask = None
                    
                else:
                    # Recursively process
                    prune_module(child, full_name)
        
        prune_module(pruned_model)
        
        self.pruned_model = pruned_model
        return pruned_model
    
    def get_params_count(self):
        """Get parameter count before and after pruning"""
        original_params = sum(p.numel() for p in self.model.parameters())
        
        if self.pruned_model is None:
            self.prune()
        
        pruned_params = sum(p.numel() for p in self.pruned_model.parameters())
        
        return original_params, pruned_params
    
    def get_flops_count(self, input_size=(1, 3, 32, 32)):
        """Estimate FLOPs before and after pruning"""
        def count_conv_flops(conv, input_shape):
            """Count FLOPs for a conv layer"""
            batch_size, in_channels, in_h, in_w = input_shape
            out_channels = conv.out_channels
            kernel_h, kernel_w = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
            
            # Output dimensions
            out_h = (in_h + 2 * conv.padding[0] - kernel_h) // conv.stride[0] + 1
            out_w = (in_w + 2 * conv.padding[1] - kernel_w) // conv.stride[1] + 1
            
            # FLOPs = output_size * kernel_size * in_channels * out_channels
            flops = out_h * out_w * kernel_h * kernel_w * in_channels * out_channels
            
            return flops, (batch_size, out_channels, out_h, out_w)
        
        def count_model_flops(model, input_shape):
            total_flops = 0
            current_shape = input_shape
            
            def traverse(module):
                nonlocal total_flops, current_shape
                
                for child in module.children():
                    if isinstance(child, nn.Conv2d):
                        flops, current_shape = count_conv_flops(child, current_shape)
                        total_flops += flops
                    elif isinstance(child, nn.Linear):
                        # FLOPs for linear layer
                        total_flops += child.in_features * child.out_features
                    else:
                        traverse(child)
            
            traverse(model)
            return total_flops
        
        original_flops = count_model_flops(self.model, input_size)
        
        if self.pruned_model is None:
            self.prune()
        
        pruned_flops = count_model_flops(self.pruned_model, input_size)
        
        return original_flops, pruned_flops
