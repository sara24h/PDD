import torch
import torch.nn as nn
import copy


class ModelPruner:
    """
    Prune the student model based on learned masks.
    Assumes masks are stored with FULL layer names (e.g., 'layer1.0.conv1').
    """
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks

    def prune(self):
        """
        Prune the model by removing channels where mask <= 0.5.
        Returns a new structurally pruned model.
        """
        pruned_model = copy.deepcopy(self.model)
        print("\nPruning model based on learned masks...")

        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.masks:
                mask = self.masks[name].squeeze()
                keep_indices = torch.nonzero(mask > 0.5).squeeze()

                if keep_indices.numel() == 0:
                    keep_indices = torch.tensor([0])
                if keep_indices.dim() == 0:
                    keep_indices = keep_indices.unsqueeze(0)

                original_channels = module.out_channels
                pruned_channels = len(keep_indices)
                print(f"{name}: {original_channels} -> {pruned_channels} channels "
                      f"({(1 - pruned_channels/original_channels)*100:.2f}% pruned)")

                self._prune_conv_layer(pruned_model, name, keep_indices)

        return pruned_model

    def _prune_conv_layer(self, model, layer_name, keep_indices):
        """Prune output channels of conv and adjust BN + next conv input."""
        modules_dict = dict(model.named_modules())
        layer = modules_dict[layer_name]

        # Prune conv output
        layer.weight = nn.Parameter(layer.weight.data[keep_indices])
        layer.out_channels = len(keep_indices)

        if layer.bias is not None:
            layer.bias = nn.Parameter(layer.bias.data[keep_indices])

        # Prune corresponding BatchNorm
        bn_name = layer_name.replace('conv', 'bn')
        if bn_name in modules_dict:
            bn = modules_dict[bn_name]
            bn.num_features = len(keep_indices)
            bn.weight = nn.Parameter(bn.weight.data[keep_indices])
            bn.bias = nn.Parameter(bn.bias.data[keep_indices])
            bn.running_mean = bn.running_mean[keep_indices]
            bn.running_var = bn.running_var[keep_indices]

        # Adjust next Conv2d input channels
        self._adjust_next_conv(model, layer_name, keep_indices)

    def _adjust_next_conv(self, model, layer_name, keep_indices):
        """Find next Conv2d and prune its input channels."""
        found = False
        for name, module in model.named_modules():
            if found and isinstance(module, nn.Conv2d):
                # Prune input channels
                new_weight = module.weight.data[:, keep_indices]
                module.weight = nn.Parameter(new_weight)
                module.in_channels = len(keep_indices)
                break
            if name == layer_name:
                found = True

    def get_params_count(self):
        """Return actual parameter counts before and after pruning."""
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_model = self.prune()
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        return original_params, pruned_params

    def get_flops_count(self, input_size=(1, 3, 32, 32)):
        """
        Return actual FLOPs before and after pruning using thop.
        Falls back to manual if thop not available.
        """
        try:
            from thop import profile
            dummy_input = torch.randn(input_size)

            # Original FLOPs
            flops_orig, _ = profile(
                copy.deepcopy(self.model),
                inputs=(dummy_input,),
                verbose=False
            )

            # Pruned FLOPs
            pruned_model = self.prune()
            flops_pruned, _ = profile(
                pruned_model,
                inputs=(dummy_input,),
                verbose=False
            )

            return int(flops_orig), int(flops_pruned)

        except ImportError:
            print("Warning: thop not installed. FLOPs may be inaccurate.")
            return self._fallback_flops(input_size)

    def _fallback_flops(self, input_size):
        """Fallback FLOPs estimation by tracing actual feature map sizes."""
        model_orig = copy.deepcopy(self.model)
        model_pruned = self.prune()

        def compute_flops(model):
            total_flops = 0
            x = torch.randn(input_size)

            def hook_fn(module, inp, out):
                nonlocal total_flops
                if isinstance(module, nn.Conv2d):
                    c_in = inp[0].shape[1]
                    c_out = out.shape[1]
                    k = module.kernel_size[0]
                    h, w = out.shape[2], out.shape[3]
                    flops = 2 * c_in * c_out * k * k * h * w
                    total_flops += flops

            hooks = []
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    hooks.append(m.register_forward_hook(hook_fn))

            with torch.no_grad():
                model(x)

            for h in hooks:
                h.remove()

            return total_flops

        return int(compute_flops(model_orig)), int(compute_flops(model_pruned))
