import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet20, ResNet56
from utils.distillation import compute_accuracy, AverageMeter
from utils.pruner import count_parameters, count_flops


def get_cifar10_testloader(batch_size=100, num_workers=4):
    """
    Prepare CIFAR10 test dataloader.
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    return testloader


def evaluate_model(model, test_loader, device, model_name):
    """
    Evaluate a single model.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc1 = compute_accuracy(outputs, targets, topk=(1,))[0]
            top1.update(acc1.item(), inputs.size(0))
    
    # Count parameters and FLOPs
    num_params = count_parameters(model)
    num_flops = count_flops(model, input_size=(1, 3, 32, 32))
    
    print(f"\nResults:")
    print(f"  Test Accuracy: {top1.avg:.2f}%")
    print(f"  Parameters: {num_params:,}")
    print(f"  FLOPs: {num_flops:,}")
    
    return top1.avg, num_params, num_flops


def main():
    parser = argparse.ArgumentParser(description='Evaluate Models')
    parser.add_argument('--teacher', type=str, default='checkpoints/resnet56_teacher_best.pth',
                       help='Path to teacher checkpoint')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline student checkpoint')
    parser.add_argument('--pruned', type=str, default='checkpoints/pruned_model_finetuned_best.pth',
                       help='Path to pruned model checkpoint')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare data
    print('Preparing test data...')
    test_loader = get_cifar10_testloader(batch_size=args.batch_size)
    
    results = {}
    
    # Evaluate Teacher
    if args.teacher:
        print('\n' + '='*60)
        print('Loading Teacher Model (ResNet56)...')
        teacher = ResNet56(num_classes=10).to(device)
        checkpoint = torch.load(args.teacher)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        
        acc, params, flops = evaluate_model(teacher, test_loader, device, "Teacher (ResNet56)")
        results['teacher'] = {'accuracy': acc, 'parameters': params, 'flops': flops}
    
    # Evaluate Baseline Student
    if args.baseline:
        print('\n' + '='*60)
        print('Loading Baseline Student Model (ResNet20)...')
        baseline = ResNet20(num_classes=10).to(device)
        checkpoint = torch.load(args.baseline)
        baseline.load_state_dict(checkpoint['model_state_dict'])
        
        acc, params, flops = evaluate_model(baseline, test_loader, device, "Baseline Student (ResNet20)")
        results['baseline'] = {'accuracy': acc, 'parameters': params, 'flops': flops}
    
    # Evaluate Pruned Model
    if args.pruned:
        print('\n' + '='*60)
        print('Loading Pruned Model (ResNet20)...')
        pruned = ResNet20(num_classes=10).to(device)
        checkpoint = torch.load(args.pruned)
        if 'model_state_dict' in checkpoint:
            pruned.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            pruned.load_state_dict(checkpoint, strict=False)
        
        acc, params, flops = evaluate_model(pruned, test_loader, device, "Pruned Model (PDD)")
        results['pruned'] = {'accuracy': acc, 'parameters': params, 'flops': flops}
    
    # Print comparison
    if len(results) > 1:
        print('\n' + '='*60)
        print('COMPARISON SUMMARY')
        print('='*60)
        
        if 'baseline' in results and 'pruned' in results:
            baseline_acc = results['baseline']['accuracy']
            pruned_acc = results['pruned']['accuracy']
            baseline_params = results['baseline']['parameters']
            pruned_params = results['pruned']['parameters']
            baseline_flops = results['baseline']['flops']
            pruned_flops = results['pruned']['flops']
            
            param_reduction = (1 - pruned_params / baseline_params) * 100
            flops_reduction = (1 - pruned_flops / baseline_flops) * 100
            acc_change = pruned_acc - baseline_acc
            
            print(f"\nBaseline Student vs. Pruned Model:")
            print(f"  Accuracy Change: {acc_change:+.2f}% ({baseline_acc:.2f}% → {pruned_acc:.2f}%)")
            print(f"  Parameter Reduction: {param_reduction:.2f}% ({baseline_params:,} → {pruned_params:,})")
            print(f"  FLOPs Reduction: {flops_reduction:.2f}% ({baseline_flops:,} → {pruned_flops:,})")
            
            print(f"\nExpected Results (from paper):")
            print(f"  Accuracy Change: +0.17%")
            print(f"  Parameter Reduction: 32.77%")
            print(f"  FLOPs Reduction: 39.53%")
    
    print('\n' + '='*60)


if __name__ == '__main__':
    main()
