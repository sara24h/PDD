import torch
import torch.nn as nn
import copy


class ModelPruner:
    """
    Prune the student model based on learned masks
    """
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        
    def prune(self):
        """
        Prune the model by removing channels with mask value 0
        """
        pruned_model = copy.deepcopy(self.model)
        
        print("\nPruning model based on learned masks...")
        
        # Prune each layer
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.masks:
                mask = self.masks[name].squeeze()
                
                # Get indices of channels to keep
                keep_indices = torch.nonzero(mask > 0.5).squeeze()
                
                if keep_indices.numel() == 0:
                    # Keep at least one channel
                    keep_indices = torch.tensor([0])
                
                # Ensure keep_indices is 1D
                if keep_indices.dim() == 0:
                    keep_indices = keep_indices.unsqueeze(0)
                
                original_channels = module.out_channels
                pruned_channels = len(keep_indices)
                
                print(f"{name}: {original_channels} -> {pruned_channels} channels "
                      f"({(1-pruned_channels/original_channels)*100:.2f}% pruned)")
                
                # Prune the conv layer
                self._prune_conv_layer(pruned_model, name, keep_indices)
        
        return pruned_model
    
    def _prune_conv_layer(self, model, layer_name, keep_indices):
        """
        Prune a specific conv layer and adjust the following layer
        """
        # Get the layer
        layer = dict(model.named_modules())[layer_name]
        
        # Prune output channels
        new_weight = layer.weight.data[keep_indices]
        layer.out_channels = len(keep_indices)
        layer.weight = nn.Parameter(new_weight)
        
        if layer.bias is not None:
            new_bias = layer.bias.data[keep_indices]
            layer.bias = nn.Parameter(new_bias)
        
        # Find and prune the corresponding BatchNorm layer
        bn_name = layer_name.replace('conv', 'bn')
        if bn_name in dict(model.named_modules()):
            bn_layer = dict(model.named_modules())[bn_name]
            
            bn_layer.num_features = len(keep_indices)
            bn_layer.weight = nn.Parameter(bn_layer.weight.data[keep_indices])
            bn_layer.bias = nn.Parameter(bn_layer.bias.data[keep_indices])
            bn_layer.running_mean = bn_layer.running_mean[keep_indices]
            bn_layer.running_var = bn_layer.running_var[keep_indices]
        
        # Adjust the next conv layer's input channels
        self._adjust_next_conv(model, layer_name, keep_indices)
    
    def _adjust_next_conv(self, model, layer_name, keep_indices):
        """
        Adjust the input channels of the next conv layer
        """
        layers = list(model.named_modules())
        current_idx = None
        
        for idx, (name, module) in enumerate(layers):
            if name == layer_name:
                current_idx = idx
                break
        
        if current_idx is None:
            return
        
        # Find next conv layer
        for idx in range(current_idx + 1, len(layers)):
            name, module = layers[idx]
            if isinstance(module, nn.Conv2d):
                # Adjust input channels
                new_weight = module.weight.data[:, keep_indices]
                module.in_channels = len(keep_indices)
                module.weight = nn.Parameter(new_weight)
                break
    
    def get_params_count(self):
        """
        Calculate number of parameters before and after pruning
        """
        original_params = sum(p.numel() for p in self.model.parameters())
        
        # Calculate pruned params
        pruned_params = 0
        for name, param in self.model.named_parameters():
            layer_name = '.'.join(name.split('.')[:-1])
            
            if layer_name in self.masks and 'weight' in name:
                mask = self.masks[layer_name].squeeze()
                keep_indices = torch.nonzero(mask > 0.5).squeeze()
                
                if keep_indices.numel() == 0:
                    keep_indices = torch.tensor([0])
                if keep_indices.dim() == 0:
                    keep_indices = keep_indices.unsqueeze(0)
                
                # Calculate pruned weight size
                if 'conv' in layer_name:
                    # Conv weight shape: [out_channels, in_channels, k, k]
                    pruned_size = len(keep_indices) * param.shape[1] * param.shape[2] * param.shape[3]
                    pruned_params += pruned_size
                else:
                    pruned_params += param.numel()
            else:
                pruned_params += param.numel()
        
        return original_params, pruned_params
    
    def get_flops_count(self, input_size=(1, 3, 32, 32)):
        """
        Estimate FLOPs before and after pruning
        """
        def count_conv_flops(layer, input_size):
            # FLOPs = 2 * C_in * C_out * K * K * H_out * W_out
            c_out = layer.out_channels
            c_in = layer.in_channels
            k = layer.kernel_size[0]
            h_out = (input_size[2] + 2 * layer.padding[0] - k) // layer.stride[0] + 1
            w_out = (input_size[3] + 2 * layer.padding[1] - k) // layer.stride[1] + 1
            flops = 2 * c_in * c_out * k * k * h_out * w_out
            return flops
        
        original_flops = 0
        pruned_flops = 0
        
        # This is a simplified estimation
        # In practice, you would need to trace through the model
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                flops = count_conv_flops(module, input_size)
                original_flops += flops
                
                if name in self.masks:
                    mask = self.masks[name].squeeze()
                    keep_ratio = (mask > 0.5).sum().item() / mask.numel()
                    pruned_flops += flops * keep_ratio
                else:
                    pruned_flops += flops
        
        return original_flops, pruned_flops
