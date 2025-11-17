from setuptools import setup, find_packages

setup(
    name='pdd-pruning',
    version='1.0.0',
    description='Pruning During Distillation for Neural Networks',
    author='PDD Implementation',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.62.0',
        'tensorboard>=2.7.0',
    ],
    python_requires='>=3.7',
)
