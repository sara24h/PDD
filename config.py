"""
Configuration file for PDD (Pruning During Distillation)
Contains all hyperparameters and settings
"""

class Config:
    """Configuration class for PDD training"""
    
    # Data settings
    DATA_DIR = './data'
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    NUM_CLASSES = 10
    
    # Model settings
    STUDENT_MODEL = 'resnet20'
    TEACHER_MODEL = 'resnet56'
    TEACHER_CHECKPOINT = 'checkpoints/resnet56_cifar10.pth'
    
    # Teacher checkpoint URL
    TEACHER_URL = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt"
    
    # Training settings - Phase 1: Distillation
    DISTILL_EPOCHS = 50
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.005
    LR_DECAY_EPOCHS = [20, 40]
    LR_DECAY_RATE = 0.1
    
    # Distillation settings
    TEMPERATURE = 4.0
    ALPHA = 0.5  # Weight for distillation loss
    
    # Training settings - Phase 3: Fine-tuning
    FINETUNE_EPOCHS = 100
    FINETUNE_LR = 0.01
    FINETUNE_MOMENTUM = 0.9
    FINETUNE_WEIGHT_DECAY = 0.005
    FINETUNE_LR_DECAY_EPOCHS = [60, 80]
    FINETUNE_LR_DECAY_RATE = 0.1
    
    # Other settings
    SEED = 42
    DEVICE = 'cuda'
    SAVE_DIR = './checkpoints'
    LOG_DIR = './logs'
    
    # Pruning settings
    MASK_INIT_METHOD = 'random'  # 'random' or 'ones'
    PRUNING_THRESHOLD = 0.5
    
    # Logging settings
    PRINT_FREQ = 50
    SAVE_FREQ = 10
    
    @classmethod
    def from_args(cls, args):
        """Update config from command line arguments"""
        config = cls()
        
        # Update from args if provided
        if hasattr(args, 'data_dir'):
            config.DATA_DIR = args.data_dir
        if hasattr(args, 'batch_size'):
            config.BATCH_SIZE = args.batch_size
        if hasattr(args, 'num_workers'):
            config.NUM_WORKERS = args.num_workers
        if hasattr(args, 'epochs'):
            config.DISTILL_EPOCHS = args.epochs
        if hasattr(args, 'lr'):
            config.LEARNING_RATE = args.lr
        if hasattr(args, 'momentum'):
            config.MOMENTUM = args.momentum
        if hasattr(args, 'weight_decay'):
            config.WEIGHT_DECAY = args.weight_decay
        if hasattr(args, 'temperature'):
            config.TEMPERATURE = args.temperature
        if hasattr(args, 'alpha'):
            config.ALPHA = args.alpha
        if hasattr(args, 'finetune_epochs'):
            config.FINETUNE_EPOCHS = args.finetune_epochs
        if hasattr(args, 'finetune_lr'):
            config.FINETUNE_LR = args.finetune_lr
        if hasattr(args, 'seed'):
            config.SEED = args.seed
        if hasattr(args, 'device'):
            config.DEVICE = args.device
        if hasattr(args, 'save_dir'):
            config.SAVE_DIR = args.save_dir
        if hasattr(args, 'log_dir'):
            config.LOG_DIR = args.log_dir
        if hasattr(args, 'teacher_checkpoint'):
            config.TEACHER_CHECKPOINT = args.teacher_checkpoint
            
        return config
    
    def display(self):
        """Display all configuration settings"""
        print("\n" + "="*60)
        print("Configuration Settings")
        print("="*60)
        
        print("\n[Data Settings]")
        print(f"Data Directory: {self.DATA_DIR}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Num Workers: {self.NUM_WORKERS}")
        print(f"Num Classes: {self.NUM_CLASSES}")
        
        print("\n[Model Settings]")
        print(f"Student Model: {self.STUDENT_MODEL}")
        print(f"Teacher Model: {self.TEACHER_MODEL}")
        print(f"Teacher Checkpoint: {self.TEACHER_CHECKPOINT}")
        
        print("\n[Phase 1: Distillation Settings]")
        print(f"Epochs: {self.DISTILL_EPOCHS}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Momentum: {self.MOMENTUM}")
        print(f"Weight Decay: {self.WEIGHT_DECAY}")
        print(f"LR Decay Epochs: {self.LR_DECAY_EPOCHS}")
        print(f"LR Decay Rate: {self.LR_DECAY_RATE}")
        print(f"Temperature: {self.TEMPERATURE}")
        print(f"Alpha: {self.ALPHA}")
        
        print("\n[Phase 3: Fine-tuning Settings]")
        print(f"Epochs: {self.FINETUNE_EPOCHS}")
        print(f"Learning Rate: {self.FINETUNE_LR}")
        print(f"Momentum: {self.FINETUNE_MOMENTUM}")
        print(f"Weight Decay: {self.FINETUNE_WEIGHT_DECAY}")
        print(f"LR Decay Epochs: {self.FINETUNE_LR_DECAY_EPOCHS}")
        print(f"LR Decay Rate: {self.FINETUNE_LR_DECAY_RATE}")
        
        print("\n[Other Settings]")
        print(f"Random Seed: {self.SEED}")
        print(f"Device: {self.DEVICE}")
        print(f"Save Directory: {self.SAVE_DIR}")
        print(f"Log Directory: {self.LOG_DIR}")
        print(f"Pruning Threshold: {self.PRUNING_THRESHOLD}")
        
        print("="*60 + "\n")


def get_default_config():
    """Get default configuration"""
    return Config()


if __name__ == '__main__':
    # Test configuration display
    config = Config()
    config.display()
