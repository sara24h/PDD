import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    """ResNet-20 for CIFAR10 (3n+2 layers, n=3)"""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    """ResNet-32 for CIFAR10 (3n+2 layers, n=5)"""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    """ResNet-44 for CIFAR10 (3n+2 layers, n=7)"""
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet50(num_classes=10):
    """ResNet-50 for CIFAR10 (3n+2 layers, n=8)"""
    return ResNet(BasicBlock, [8, 8, 8], num_classes=num_classes)


def resnet56(num_classes=10):
    """ResNet-56 for CIFAR10 (3n+2 layers, n=9)"""
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(num_classes=10):
    """ResNet-110 for CIFAR10 (3n+2 layers, n=18)"""
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet164(num_classes=10):
    """ResNet-164 for CIFAR10 (3n+2 layers, n=27)"""
    return ResNet(BasicBlock, [27, 27, 27], num_classes=num_classes)


def resnet1202(num_classes=10):
    """ResNet-1202 for CIFAR10 (3n+2 layers, n=200)"""
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)


if __name__ == '__main__':
    # Test models
    def test_model(model, name):
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:15s} - Output: {y.shape}, Parameters: {params:,}")
    
    print("Testing CIFAR10 ResNet models:")
    print("-" * 60)
    test_model(resnet20(), "ResNet-20")
    test_model(resnet32(), "ResNet-32")
    test_model(resnet44(), "ResNet-44")
    test_model(resnet50(), "ResNet-50")
    test_model(resnet56(), "ResNet-56")
    test_model(resnet110(), "ResNet-110")
    test_model(resnet164(), "ResNet-164")
