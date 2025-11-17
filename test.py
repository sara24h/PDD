import torch
import argparse
from utils.data_loader import get_cifar10_dataloaders
from models.resnet import resnet20
from utils.helpers import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Test PDD Pruned Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def test_model(model, test_loader, device):
    """Test the model and return accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading CIFAR10 test dataset...")
    _, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    model = resnet20(num_classes=10)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model)
    
    if checkpoint is not None:
        print("\nCheckpoint Info:")
        if 'accuracy' in checkpoint:
            print(f"Saved Accuracy: {checkpoint['accuracy']:.2f}%")
        if 'params_reduction' in checkpoint:
            print(f"Parameters Reduction: {checkpoint['params_reduction']:.2f}%")
        if 'flops_reduction' in checkpoint:
            print(f"FLOPs Reduction: {checkpoint['flops_reduction']:.2f}%")
    
    model = model.to(device)
    
    # Test model
    print("\nTesting model...")
    accuracy = test_model(model, test_loader, device)
    
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")


if __name__ == '__main__':
    main()
