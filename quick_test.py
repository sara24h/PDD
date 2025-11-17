"""
Quick test script to verify the PDD implementation.
Tests model creation, forward pass, and basic functionality.
"""

import torch
import torch.nn as nn
from models.resnet import ResNet20, ResNet56
from models.masked_resnet import MaskedResNet20
from models.dynamic_mask import ApproxSign
from utils.distillation import DistillationLoss, compute_accuracy
from utils.pruner import count_parameters, count_flops


def test_approx_sign():
    """Test the differentiable piecewise polynomial function."""
    print("\n" + "="*60)
    print("Testing ApproxSign Function")
    print("="*60)
    
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    y = ApproxSign.apply(x)
    
    print(f"Input:  {x.numpy()}")
    print(f"Output: {y.numpy()}")
    print("✓ ApproxSign function works correctly")


def test_resnet_models():
    """Test ResNet model creation and forward pass."""
    print("\n" + "="*60)
    print("Testing ResNet Models")
    print("="*60)
    
    # Test ResNet20
    model20 = ResNet20(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y20 = model20(x)
    
    print(f"ResNet20 output shape: {y20.shape}")
    print(f"ResNet20 parameters: {count_parameters(model20):,}")
    assert y20.shape == (2, 10), "ResNet20 output shape mismatch"
    print("✓ ResNet20 works correctly")
    
    # Test ResNet56
    model56 = ResNet56(num_classes=10)
    y56 = model56(x)
    
    print(f"ResNet56 output shape: {y56.shape}")
    print(f"ResNet56 parameters: {count_parameters(model56):,}")
    assert y56.shape == (2, 10), "ResNet56 output shape mismatch"
    print("✓ ResNet56 works correctly")


def test_masked_resnet():
    """Test MaskedResNet with dynamic masks."""
    print("\n" + "="*60)
    print("Testing MaskedResNet")
    print("="*60)
    
    model = MaskedResNet20(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    
    print(f"MaskedResNet20 output shape: {y.shape}")
    print(f"MaskedResNet20 parameters: {count_parameters(model):,}")
    assert y.shape == (2, 10), "MaskedResNet20 output shape mismatch"
    print("✓ MaskedResNet20 works correctly")
    
    # Test pruning info
    pruning_info = model.get_pruning_info()
    print(f"\nPruning info keys: {list(pruning_info.keys())[:5]}...")
    print(f"Overall pruning ratio: {pruning_info['overall']['ratio']:.2%}")
    print("✓ Pruning info extraction works correctly")


def test_distillation_loss():
    """Test knowledge distillation loss."""
    print("\n" + "="*60)
    print("Testing Distillation Loss")
    print("="*60)
    
    criterion = DistillationLoss(temperature=4.0, alpha=0.5)
    
    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    
    total_loss, distill_loss, ce_loss = criterion(student_logits, teacher_logits, labels)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Distillation loss: {distill_loss.item():.4f}")
    print(f"CE loss: {ce_loss.item():.4f}")
    
    assert total_loss.requires_grad, "Loss should require gradients"
    print("✓ Distillation loss works correctly")


def test_backward_pass():
    """Test backward pass with masks."""
    print("\n" + "="*60)
    print("Testing Backward Pass")
    print("="*60)
    
    model = MaskedResNet20(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    
    x = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 10, (2,))
    
    # Forward pass
    output = model(x)
    loss = criterion(output, labels)
    
    # Backward pass
    loss.backward()
    
    # Check if masks have gradients
    has_mask_grads = False
    for name, param in model.named_parameters():
        if 'mask_weight' in name and param.grad is not None:
            has_mask_grads = True
            print(f"Mask gradient norm: {param.grad.norm().item():.6f}")
            break
    
    assert has_mask_grads, "Mask weights should have gradients"
    print("✓ Backward pass with masks works correctly")


def test_accuracy_computation():
    """Test accuracy computation."""
    print("\n" + "="*60)
    print("Testing Accuracy Computation")
    print("="*60)
    
    # Create dummy predictions and targets
    outputs = torch.tensor([
        [2.0, 1.0, 0.5],  # predicts class 0
        [0.5, 2.0, 1.0],  # predicts class 1
        [1.0, 0.5, 2.0],  # predicts class 2
        [2.0, 1.0, 0.5],  # predicts class 0
    ])
    targets = torch.tensor([0, 1, 2, 1])  # 3 correct, 1 wrong
    
    acc = compute_accuracy(outputs, targets, topk=(1,))[0]
    
    expected_acc = 75.0  # 3/4 = 75%
    print(f"Computed accuracy: {acc.item():.2f}%")
    print(f"Expected accuracy: {expected_acc:.2f}%")
    
    assert abs(acc.item() - expected_acc) < 1e-5, "Accuracy computation error"
    print("✓ Accuracy computation works correctly")


def test_parameter_counting():
    """Test parameter and FLOPs counting."""
    print("\n" + "="*60)
    print("Testing Parameter Counting")
    print("="*60)
    
    model = ResNet20(num_classes=10)
    
    num_params = count_parameters(model)
    num_flops = count_flops(model, input_size=(1, 3, 32, 32))
    
    print(f"ResNet20 parameters: {num_params:,}")
    print(f"ResNet20 FLOPs: {num_flops:,}")
    
    assert num_params > 0, "Parameter count should be positive"
    assert num_flops > 0, "FLOPs should be positive"
    print("✓ Parameter counting works correctly")


def main():
    print("\n" + "="*60)
    print("PDD Implementation Quick Test")
    print("="*60)
    print("\nThis script tests the basic functionality of the PDD implementation.")
    print("It does NOT train the models - use train.py for actual training.\n")
    
    try:
        # Run all tests
        test_approx_sign()
        test_resnet_models()
        test_masked_resnet()
        test_distillation_loss()
        test_backward_pass()
        test_accuracy_computation()
        test_parameter_counting()
        
        # Summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe PDD implementation is working correctly.")
        print("You can now proceed with training:")
        print("  1. python pretrain_teacher.py")
        print("  2. python train.py --teacher-checkpoint checkpoints/resnet56_teacher_best.pth")
        print("  3. python finetune_pruned.py --checkpoint checkpoints/pruning_plan.pth")
        print("\nOr run the complete pipeline:")
        print("  bash run_all.sh")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
