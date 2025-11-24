# utils/pruner.py

import torch
import torch.nn as nn
from thop import profile

class ModelPruner:
    def __init__(self, model, masks):
        
        self.model = model
        self.masks = masks
        self._original_params = None
        self._pruned_params = None
        self._original_flops = None
        self._pruned_flops = None

    def _calculate_flops(self, model):

        input_tensor = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return flops

    def prune(self):
      
        print("\nAnalyzing masks...")
        
        keep_indices = {}
        pruning_stats = {}
        
        for name, mask in self.masks.items():
            # ماسک از trainer به صورت binary است: 1=keep, 0=prune
            mask_flat = mask.squeeze().cpu()
            
            # ✅ کانال‌هایی که mask=1 دارند را نگه می‌داریم
            keep_idx = torch.where(mask_flat == 1.0)[0]
            
            total_channels = mask_flat.numel()
            kept_channels = len(keep_idx)
            
            keep_indices[name] = keep_idx
            pruning_stats[name] = {
                'total': total_channels,
                'kept': kept_channels,
                'pruned': total_channels - kept_channels,
                'ratio': (1 - kept_channels / total_channels) * 100
            }
            
            print(f"{name:40s} | Total: {total_channels:4d} | "
                  f"Kept: {kept_channels:4d} | "
                  f"Pruned: {pruning_stats[name]['ratio']:.2f}%")
        
        print("\nBuilding pruned model...")
        pruned_model = self._build_pruned_model(keep_indices, pruning_stats)
        
        print("✓ Pruned model created")
        
        print("\nCopying weights...")
        
        device = next(self.model.parameters()).device
        pruned_model = pruned_model.to(device)
        
        self._copy_weights(pruned_model, keep_indices)
        print("✓ Weights copied")
        
        self._calculate_compression_stats(pruned_model)
        
        return pruned_model

    def _build_pruned_model(self, keep_indices, stats):
        """ساخت مدل با کانال‌های کمتر"""
        from models.resnet import ResNet, BasicBlock
        
        # ✅ اصلاح شد: از fc استفاده می‌کنیم
        num_classes = self.model.fc.out_features
        conv1_channels = len(keep_indices['conv1']) if 'conv1' in keep_indices else 16
        
        pruned_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        
        # Update conv1
        if 'conv1' in keep_indices:
            kept = len(keep_indices['conv1'])
            pruned_model.conv1 = nn.Conv2d(3, kept, kernel_size=3, 
                                           stride=1, padding=1, bias=False)
            pruned_model.bn1 = nn.BatchNorm2d(kept)
        
        prev_channels = conv1_channels
        
        # Update each stage
        for stage_idx, stage_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            stage = getattr(pruned_model, stage_name)
            for block_idx in range(len(stage)):
                block = stage[block_idx]
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                in_channels = prev_channels
                
                conv1_name = f'{stage_name}.{block_idx}.conv1'
                conv2_name = f'{stage_name}.{block_idx}.conv2'
                
                conv1_out = len(keep_indices[conv1_name]) if conv1_name in keep_indices else block.conv1.out_channels
                conv2_out = len(keep_indices[conv2_name]) if conv2_name in keep_indices else block.conv2.out_channels
                
                block.conv1 = nn.Conv2d(in_channels, conv1_out, 
                                       kernel_size=3, stride=stride, 
                                       padding=1, bias=False)
                block.bn1 = nn.BatchNorm2d(conv1_out)
                
                block.conv2 = nn.Conv2d(conv1_out, conv2_out, 
                                       kernel_size=3, stride=1, 
                                       padding=1, bias=False)
                block.bn2 = nn.BatchNorm2d(conv2_out)
                
                # Update shortcut if needed
                if stride != 1 or in_channels != conv2_out:
                    block.downsample = nn.Sequential( # ✅ اصلاح شد: به downsample تغییر یافت
                        nn.Conv2d(in_channels, conv2_out, 
                                 kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(conv2_out)
                    )
                else:
                    block.downsample = nn.Sequential() # ✅ اصلاح شد: به downsample تغییر یافت
                
                prev_channels = conv2_out
        
        # ✅ اصلاح شد: از fc استفاده می‌کنیم
        pruned_model.fc = nn.Linear(prev_channels, num_classes)
        
        return pruned_model

    def _copy_weights(self, pruned_model, keep_indices):
        """کپی کردن وزن‌ها از مدل اصلی به مدل هرس شده"""
        
        # Conv1
        if 'conv1' in keep_indices:
            idx = keep_indices['conv1']
            pruned_model.conv1.weight.data = self.model.conv1.weight.data[idx, :, :, :]
            pruned_model.bn1.weight.data = self.model.bn1.weight.data[idx]
            pruned_model.bn1.bias.data = self.model.bn1.bias.data[idx]
            pruned_model.bn1.running_mean.data = self.model.bn1.running_mean.data[idx]
            pruned_model.bn1.running_var.data = self.model.bn1.running_var.data[idx]
        
        # Process each stage
        for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            orig_stage = getattr(self.model, stage_name)
            pruned_stage = getattr(pruned_model, stage_name)
            
            for block_idx in range(len(orig_stage)):
                orig_block = orig_stage[block_idx]
                pruned_block = pruned_stage[block_idx]
                
                # Conv1 of block
                conv1_name = f'{stage_name}.{block_idx}.conv1'
                if conv1_name in keep_indices:
                    out_idx = keep_indices[conv1_name]
                    
                    # Determine input indices
                    if block_idx == 0:
                        if stage_name == 'layer1':
                            in_idx = keep_indices.get('conv1', torch.arange(orig_block.conv1.in_channels))
                        else:
                            prev_stage = 'layer1' if stage_name == 'layer2' else 'layer2'
                            prev_conv_name = f'{prev_stage}.1.conv2'
                            in_idx = keep_indices.get(prev_conv_name, torch.arange(orig_block.conv1.in_channels))
                    else:
                        prev_conv_name = f'{stage_name}.{block_idx-1}.conv2'
                        in_idx = keep_indices.get(prev_conv_name, torch.arange(orig_block.conv1.in_channels))
                    
                    pruned_block.conv1.weight.data = orig_block.conv1.weight.data[out_idx, :, :, :][:, in_idx, :, :]
                    pruned_block.bn1.weight.data = orig_block.bn1.weight.data[out_idx]
                    pruned_block.bn1.bias.data = orig_block.bn1.bias.data[out_idx]
                    pruned_block.bn1.running_mean.data = orig_block.bn1.running_mean.data[out_idx]
                    pruned_block.bn1.running_var.data = orig_block.bn1.running_var.data[out_idx]
                
                # Conv2 of block
                conv2_name = f'{stage_name}.{block_idx}.conv2'
                if conv2_name in keep_indices:
                    out_idx = keep_indices[conv2_name]
                    in_idx = keep_indices.get(conv1_name, torch.arange(orig_block.conv2.in_channels))
                    
                    pruned_block.conv2.weight.data = orig_block.conv2.weight.data[out_idx, :, :, :][:, in_idx, :, :]
                    pruned_block.bn2.weight.data = orig_block.bn2.weight.data[out_idx]
                    pruned_block.bn2.bias.data = orig_block.bn2.bias.data[out_idx]
                    pruned_block.bn2.running_mean.data = orig_block.bn2.running_mean.data[out_idx]
                    pruned_block.bn2.running_var.data = orig_block.bn2.running_var.data[out_idx]
                
                # ✅ اصلاح شد: از downsample استفاده می‌کنیم
                # Shortcut
                if orig_block.downsample is not None and len(orig_block.downsample) > 0:
                    for i, layer in enumerate(orig_block.downsample):
                        if isinstance(layer, nn.Conv2d):
                            if block_idx == 0:
                                if stage_name == 'layer1':
                                    in_idx = keep_indices.get('conv1', torch.arange(layer.in_channels))
                                else:
                                    prev_stage = 'layer1' if stage_name == 'layer2' else 'layer2'
                                    prev_conv_name = f'{prev_stage}.1.conv2'
                                    in_idx = keep_indices.get(prev_conv_name, torch.arange(layer.in_channels))
                            else:
                                prev_conv_name = f'{stage_name}.{block_idx-1}.conv2'
                                in_idx = keep_indices.get(prev_conv_name, torch.arange(layer.in_channels))
                            
                            out_idx = keep_indices.get(conv2_name, torch.arange(layer.out_channels))
                            pruned_block.downsample[i].weight.data = layer.weight.data[out_idx, :, :, :][:, in_idx, :, :]
                        
                        elif isinstance(layer, nn.BatchNorm2d):
                            out_idx = keep_indices.get(conv2_name, torch.arange(layer.num_features))
                            pruned_block.downsample[i].weight.data = layer.weight.data[out_idx]
                            pruned_block.downsample[i].bias.data = layer.bias.data[out_idx]
                            pruned_block.downsample[i].running_mean.data = layer.running_mean.data[out_idx]
                            pruned_block.downsample[i].running_var.data = layer.running_var.data[out_idx]
        
        # ✅ اصلاح شد: از fc استفاده می‌کنیم
        last_stage_last_block = f'layer4.1.conv2'
        if last_stage_last_block in keep_indices:
            in_idx = keep_indices[last_stage_last_block]
            pruned_model.fc.weight.data = self.model.fc.weight.data[:, in_idx]
            pruned_model.fc.bias.data = self.model.fc.bias.data

    def _calculate_compression_stats(self, pruned_model):
        """محاسبه آمار فشرده‌سازی"""
        self._original_params = sum(p.numel() for p in self.model.parameters())
        self._pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        self._original_flops = self._calculate_flops(self.model)
        self._pruned_flops = self._calculate_flops(pruned_model)

    def get_params_count(self):
        return self._original_params, self._pruned_params

    def get_flops_count(self):
        return self._original_flops, self._pruned_flops
