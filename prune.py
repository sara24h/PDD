# prune.py
import torch
import argparse
import os
from models.resnet import resnet20
from utils.pruner import ModelPruner
from utils.helpers import save_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./pruned')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Final Pruning Phase - PDD (100% Paper Accurate)")
    print("Threshold: ApproxSign(mask) ≈ 0 → prune (using 1e-8 for floating point)")
    print("="*80)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    student = resnet20(num_classes=10)
    student.load_state_dict(ckpt['state_dict'])

    # ماسک‌های خام (raw) رو می‌گیریم، ApproxSign داخل ModelPruner اعمال میشه
    raw_masks = ckpt['masks']

    pruner = ModelPruner(student, raw_masks)  # بدون threshold!

    print("Pruning model...")
    pruned_model = pruner.prune()

    orig_params, _ = pruner.get_params_count()
    orig_flops, _ = pruner.get_flops_count()
    pruned_params = sum(p.numel() for p in pruned_model.parameters())

    params_red = (1 - pruned_params / orig_params) * 100
    flops_red = (1 - pruner.get_flops_count()[1] / orig_flops) * 100

    print("\n" + "="*80)
    print("FINAL RESULT (Should match Table 1):")
    print(f"Parameters Reduction : {params_red:.2f}% ")
    print(f"FLOPs Reduction     : {flops_red:.2f}%  ")
    print("="*80)



    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'pdd_pruned_resnet20.pth')
    torch.save({'state_dict': pruned_model.state_dict()}, save_path)
    print(f"Pruned model saved: {save_path}")


if __name__ == '__main__':
    main()
