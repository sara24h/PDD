"""
Models package for PDD (Pruning During Distillation)
Contains ResNet architectures for CIFAR10
"""

from .resnet import resnet20, resnet56, resnet110, ResNet, BasicBlock

__all__ = [
    'resnet20',
    'resnet56', 
    'resnet110',
    'ResNet',
    'BasicBlock'
]

__version__ = '1.0.0'
