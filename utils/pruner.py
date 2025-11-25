# utils/pruner.py

import torch
import torch.nn as nn
from thop import profile

class ModelPruner:
    def __init__(self, model, masks):
        """
        مقداردهی اولیه کلاس Pruner
        
        Args:
            model (nn.Module): مدل اصلی که هرس می‌شود.
            masks (dict): دیکشنری از ماسک‌های باینری برای هر لایه.
        """
        self.model = model
        self.masks = masks
        self._original_params = None
        self._pruned_params = None
        self._original_flops = None
        self._pruned_flops = None

    def _calculate_flops(self, model):
        """محاسبه تعداد عملیات مم شناور (FLOPS) برای یک مدل."""
        # یک تانسور ورودی نمونه برای محاسبه FLOPS
        input_tensor = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return flops

    def prune(self):
        """
        متد اصلی برای انجام هرس کردن مدل.
        1. تحلیل ماسک‌ها
        2. ساخت مدل هرس شده
        3. کپی کردن وزن‌ها
        4. محاسبه آمار فشرده‌سازی
        """
        print("\nAnalyzing masks...")
        
        keep_indices = {}
        pruning_stats = {}
        
        for name, mask in self.masks.items():
            # ماسک از trainer به صورت binary است: 1=keep, 0=prune
            mask_flat = mask.squeeze().cpu()
            
            # کانال‌هایی که mask=1 دارند را نگه می‌داریم
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
        self._copy_weights(pruned_model, keep_indices)
        print("✓ Weights copied")
        
        # انتقال مدل هرس شده به دستگاه (GPU/CPU) مدل اصلی برای جلوگیری از خطا
        device = next(self.model.parameters()).device
        pruned_model = pruned_model.to(device)
        
        self._calculate_compression_stats(pruned_model)
        
        return pruned_model

    def _build_pruned_model(self, keep_indices, stats):
        """ساخت مدل با کانال‌های کمتر بر اساس ماسک‌ها."""
        from models.resnet import ResNet, BasicBlock
        
        num_classes = self.model.fc.out_features
        
        # ساخت یک مدل ResNet18 جدید از پایه
        pruned_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        
        # --- به‌روزرسانی لایه‌های اولیه ---
        # لایه conv1
        if 'conv1' in keep_indices:
            kept_channels = len(keep_indices['conv1'])
            pruned_model.conv1 = nn.Conv2d(3, kept_channels, kernel_size=7, stride=2, padding=3, bias=False)
            pruned_model.bn1 = nn.BatchNorm2d(kept_channels)
        
        # --- به‌روزرسانی لایه‌های اصلی (Stages) ---
        # این متغیر تعداد کانال‌های خروجی لایه قبلی را نگه می‌دارد
        prev_out_channels = len(keep_indices['conv1'])

        stage_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        for i, stage_name in enumerate(stage_names):
            pruned_stage = getattr(pruned_model, stage_name)
            
            for j, block in enumerate(pruned_stage):
                # نام لایه‌های کانولوشنی در این بلوک
                conv1_name = f'{stage_name}.{j}.conv1'
                conv2_name = f'{stage_name}.{j}.conv2'
                
                # تعداد کانال‌های خروجی برای هر کانولوشن بر اساس ماسک
                conv1_out_channels = len(keep_indices[conv1_name])
                conv2_out_channels = len(keep_indices[conv2_name])
                
                # اگر اولین بلوک از استیج‌های بعدی باشد، باید stride=2 باشد
                stride = 2 if (j == 0 and i > 0) else 1
                
                # --- ساخت مجدد لایه‌های بلوک ---
                # لایه اول کانولوشن
                block.conv1 = nn.Conv2d(prev_out_channels, conv1_out_channels, 
                                       kernel_size=3, stride=stride, padding=1, bias=False)
                block.bn1 = nn.BatchNorm2d(conv1_out_channels)
                
                # لایه دوم کانولوشن
                block.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, 
                                       kernel_size=3, stride=1, padding=1, bias=False)
                block.bn2 = nn.BatchNorm2d(conv2_out_channels)
                
                if stride != 1 or prev_out_channels != conv2_out_channels:
                    block.downsample = nn.Sequential(
                        nn.Conv2d(prev_out_channels, conv2_out_channels, 
                                 kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(conv2_out_channels)
                    )
                else:
                    block.downsample = nn.Sequential()
                
                prev_out_channels = conv2_out_channels
        

        pruned_model.fc = nn.Linear(prev_out_channels, 1)  
        
        return pruned_model

    def _copy_weights(self, pruned_model, keep_indices):
      
        if 'conv1' in keep_indices:
            keep_idx = keep_indices['conv1']
            pruned_model.conv1.weight.data = self.model.conv1.weight.data[keep_idx, :, :, :]
            pruned_model.bn1.weight.data = self.model.bn1.weight.data[keep_idx]
            pruned_model.bn1.bias.data = self.model.bn1.bias.data[keep_idx]
            pruned_model.bn1.running_mean.data = self.model.bn1.running_mean.data[keep_idx]
            pruned_model.bn1.running_var.data = self.model.bn1.running_var.data[keep_idx]

        stage_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        for i, stage_name in enumerate(stage_names):
            orig_stage = getattr(self.model, stage_name)
            pruned_stage = getattr(pruned_model, stage_name)
            
            for j, pruned_block in enumerate(pruned_stage):
                orig_block = orig_stage[j]
                
                conv1_name = f'{stage_name}.{j}.conv1'
                conv2_name = f'{stage_name}.{j}.conv2'
                
                # --- تعیین ایندکس کانال‌های ورودی برای conv1 ---
                # کانال‌های ورودی conv1 برابر با کانال‌های خروجی بلوک قبلی است
                if j == 0:
                    if i == 0: # اولین بلوک از layer1، ورودی از conv1 می‌آید
                        in_idx = keep_indices['conv1']
                    else: # اولین بلوک از استیج‌های دیگر، ورودی از آخرین بلوک استیج قبلی می‌آید
                        prev_stage_name = stage_names[i-1]
                        prev_conv2_name = f'{prev_stage_name}.1.conv2' # فرض بر این است که هر استیج 2 بلوک دارد
                        in_idx = keep_indices.get(prev_conv2_name, torch.arange(orig_block.conv1.in_channels))
                else: # بلوک‌های میانی، ورودی از بلوک قبلی در همان استیج می‌آید
                    prev_conv2_name = f'{stage_name}.{j-1}.conv2'
                    in_idx = keep_indices.get(prev_conv2_name, torch.arange(orig_block.conv1.in_channels))

                # --- کپی کردن وزن‌های conv1 و bn1 ---
                if conv1_name in keep_indices:
                    out_idx = keep_indices[conv1_name]
                    pruned_block.conv1.weight.data = orig_block.conv1.weight.data[out_idx][:, in_idx, :, :]
                    pruned_block.bn1.weight.data = orig_block.bn1.weight.data[out_idx]
                    pruned_block.bn1.bias.data = orig_block.bn1.bias.data[out_idx]
                    pruned_block.bn1.running_mean.data = orig_block.bn1.running_mean.data[out_idx]
                    pruned_block.bn1.running_var.data = orig_block.bn1.running_var.data[out_idx]

                # --- کپی کردن وزن‌های conv2 و bn2 ---
                if conv2_name in keep_indices:
                    out_idx = keep_indices[conv2_name]
                    # کانال‌های ورودی conv2 برابر با کانال‌های خروجی conv1 همین بلوک است
                    in_idx_conv2 = keep_indices.get(conv1_name, torch.arange(orig_block.conv2.in_channels))
                    
                    pruned_block.conv2.weight.data = orig_block.conv2.weight.data[out_idx][:, in_idx_conv2, :, :]
                    pruned_block.bn2.weight.data = orig_block.bn2.weight.data[out_idx]
                    pruned_block.bn2.bias.data = orig_block.bn2.bias.data[out_idx]
                    pruned_block.bn2.running_mean.data = orig_block.bn2.running_mean.data[out_idx]
                    pruned_block.bn2.running_var.data = orig_block.bn2.running_var.data[out_idx]

                # --- کپی کردن وزن‌های لایه downsample ---
                if orig_block.downsample is not None and len(orig_block.downsample) > 0:
                    out_idx_downsample = keep_indices.get(conv2_name, torch.arange(orig_block.downsample[0].out_channels))
                    # کانال‌های ورودی downsample همان کانال‌های ورودی conv1 است
                    in_idx_downsample = in_idx

                    pruned_block.downsample[0].weight.data = orig_block.downsample[0].weight.data[out_idx_downsample][:, in_idx_downsample, :, :]
                    pruned_block.downsample[1].weight.data = orig_block.downsample[1].weight.data[out_idx_downsample]
                    pruned_block.downsample[1].bias.data = orig_block.downsample[1].bias.data[out_idx_downsample]
                    pruned_block.downsample[1].running_mean.data = orig_block.downsample[1].running_mean.data[out_idx_downsample]
                    pruned_block.downsample[1].running_var.data = orig_block.downsample[1].running_var.data[out_idx_downsample]

    
        last_conv2_name = 'layer4.1.conv2'
        if last_conv2_name in keep_indices:
            in_idx_fc = keep_indices[last_conv2_name]
            pruned_model.fc.weight.data = self.model.fc.weight.data[:, in_idx_fc]  # [1, in_features]
            pruned_model.fc.bias.data = self.model.fc.bias.data  # [1]

    def _calculate_compression_stats(self, pruned_model):
        """محاسبه آمار فشرده‌سازی (تعداد پارامترها و FLOPS)."""
        self._original_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        
        self._original_flops = self._calculate_flops(self.model)
        self._pruned_flops = self._calculate_flops(pruned_model)

    def get_params_count(self):
        """برگرداندن تعداد پارامترهای مدل اصلی و هرس شده."""
        return self._original_params, self._pruned_params

    def get_flops_count(self):
        """برگرداندن تعداد FLOPS مدل اصلی و هرس شده."""
        return self._original_flops, self._pruned_flops
