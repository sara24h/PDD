class PDDTrainer:
    def __init__(self, student, teacher, train_loader, test_loader, device, args, rank=0):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.rank = rank  

        self.masks = self._initialize_masks()
        mask_params = list(self.masks.values())
        
        self.optimizer = torch.optim.SGD(
            [
                {'params': self.student.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': mask_params, 'lr': args.lr * 5, 'weight_decay': 0.0}
            ],
            momentum=args.momentum
        )
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_acc = 0.0
        self.best_masks = None

    def _initialize_masks(self):
        masks = {}
        # این یک پیاده‌سازی ساده برای ماسک‌ها است. در ResNet واقعی، باید لایه‌ها را پیمایش کنید.
        # برای مدل ساختگی ما، فقط یک ماسک برای لایه کانولوشنی در نظر می‌گیریم.
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = nn.Parameter(
                    torch.randn(1, module.out_channels, 1, 1, device=self.device) - 1.2,
                    requires_grad=True
                )
                masks[name] = mask
        return masks

    def _approx_sign(self, x):
        """ApproxSign function from paper"""
        return torch.where(x < -1, torch.zeros_like(x),
                           torch.where(x < 0, (x + 1)**2 / 2,
                                       torch.where(x < 1, (2*x - x**2 + 1)/2, 
                                                   torch.ones_like(x))))

    def _forward_with_masks(self, x):
        # این یک فوروارد ساده‌شده برای مدل ساختگی است.
        out = self.student[0](x) # Conv2d
        # اعمال ماسک (اگر وجود داشته باشد)
        if '0' in self.masks: # نام لایه در مدل ساختگی '0' است
            mask = self._approx_sign(self.masks['0'])
            out = out * mask
        
        out = self.student[1](out) # BatchNorm
        out = self.student[2](out) # ReLU
        out = self.student[3](out) # AdaptiveAvgPool2d
        out = self.student[4](out) # Flatten
        out = self.student[5](out) # Linear
        return out

    def train(self):
        if self.rank == 0:
            print("\n" + "="*70)
            print("PHASE 1: Pruning During Distillation")
            print("="*70)
        
        # شروع زمان‌سنجی کل فرآیند دیستیلیشن
        total_start_time = time.time()
        
        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            self.student.train()
            
            train_loss = 0.0
            kd_loss_total = 0.0
            ce_loss_total = 0.0
            correct = 0
            total = 0
            
            # ایجاد نوار پیشرفت فقط برای رنک 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}", leave=False, disable=self.rank != 0)
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device).float().unsqueeze(1)
                
                self.optimizer.zero_grad()
                
                # خروجی دانش‌آموز با ماسک‌ها
                student_logits = self._forward_with_masks(inputs)
                
                # خروجی معلم (بدون گرادینت)
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)

                # 1. زیان طبقه‌بندی (BCEWithLogitsLoss)
                ce_loss = self.criterion(student_logits, targets)
                
                # 2. زیان دیستیلیشن دانش (KD)
                teacher_probs = torch.sigmoid(teacher_logits / self.args.temperature)
                student_probs = torch.sigmoid(student_logits / self.args.temperature)
                kd_loss = F.binary_cross_entropy(student_probs, teacher_probs, reduction='mean') * (self.args.temperature ** 2)

                total_loss = self.args.alpha * kd_loss + (1 - self.args.alpha) * ce_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
                kd_loss_total += kd_loss.item()
                ce_loss_total += ce_loss.item()
                
                predicted = (student_logits > 0).float()
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # به‌روزرسانی نوار پیشرفت با متریک‌های لحظه‌ای
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            train_acc = 100. * correct / total
            test_acc = self.evaluate()
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            if self.rank == 0:
                print(f"\nEpoch [{epoch+1}/{self.args.epochs}] ({epoch_time:.2f}s)")
                print(f"Train: Loss={train_loss/len(self.train_loader):.4f}, Acc={train_acc:.2f}%")
                print(f"Test: Acc={test_acc:.2f}%")
                print(f"Losses: KD={kd_loss_total/len(self.train_loader):.4f}, CE={ce_loss_total/len(self.train_loader):.4f}")
                
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    self.best_masks = {name: mask.clone().detach() for name, mask in self.masks.items()}
                    print(f"✓ New best accuracy: {test_acc:.2f}%")
        
        if self.best_masks is not None:
            for name in self.masks.keys():
                self.masks[name].data = self.best_masks[name].data
        
        total_time = time.time() - total_start_time
        if self.rank == 0:
            print("\n" + "="*70)
            print("Distillation Training Complete!")
            print(f"Best Accuracy: {self.best_acc:.2f}%")
            print(f"Total Distillation Time: {total_time/60:.2f} minutes")
            print("="*70 + "\n")

    def evaluate(self):
        self.student.eval()
        correct = 0
        total = 0
        
        # ایجاد نوار پیشرفت برای ارزیابی
        eval_pbar = tqdm(self.test_loader, desc="Evaluating", leave=False, disable=self.rank != 0)
        
        with torch.no_grad():
            for inputs, targets in eval_pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._forward_with_masks(inputs)
                predicted = (outputs > 0).float()
                targets = targets.float().unsqueeze(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # به‌روزرسانی نوار پیشرفت ارزیابی
                eval_pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
        
        return 100. * correct / total

    def get_masks(self):
        """Generate final binary masks for pruning"""
        binary_masks = {}
        for name, mask in self.masks.items():
            score = self._approx_sign(mask).detach().squeeze()
            binary_mask = (score > 0.0).float()
            binary_masks[name] = binary_mask
        return binary_masks


# ===================================================================
# تابع اصلی اجرا برای هر پردازش (در حالت DDP)
# ===================================================================
def main_worker(rank, args):
    # Setup DDP
    if args.world_size > 1:
        setup(rank, args.world_size)
    
    set_seed(args.seed + rank)
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 1
    
    if rank == 0:
        print("\n" + "="*70)
        print("PDD: Pruning During Distillation")
        print("="*70)
        print(f"Device: {device}")
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Batch Size: {args.batch_size}")
        print("="*70)
    
    # Load data
    train_loader, test_loader = load_dataset(args, rank)
    
    # Create models
    student = resnet18(num_classes=NUM_CLASSES).to(device)
    teacher = resnet50(num_classes=NUM_CLASSES).to(device)
    
    # Wrap models with DDP
    if args.world_size > 1:
        student = DDP(student, device_ids=[rank])
        teacher = DDP(teacher, device_ids=[rank])

    if rank == 0:
        print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
        print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # Load teacher
    if rank == 0:
        print("\nLoading teacher model...")
        load_teacher_model(teacher.module if args.world_size > 1 else teacher, args.teacher_checkpoint, device)
    
    # Evaluate teacher (only on rank 0)
    if rank == 0:
        print("\nEvaluating teacher model...")
        correct = 0
        total = 0
        teacher_model = teacher.module if args.world_size > 1 else teacher
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating teacher", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = teacher_model(inputs).squeeze(1)
                preds = (outputs > 0).long()
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
        
        teacher_acc = 100. * correct / total
        print(f"Teacher Test Accuracy: {teacher_acc:.2f}%")
    else:
        teacher_acc = 0.0 # مقداردهی اولیه برای رنک‌های دیگر

    # Phase 1: PDD (Pruning During Distillation)
    distillation_start_time = time.time()
    trainer = PDDTrainer(student, teacher, train_loader, test_loader, device, args, rank=rank)
    trainer.train()
    distillation_time = time.time() - distillation_start_time

    # Save student with masks (only on rank 0)
    if rank == 0:
        checkpoint_name = f'student_resnet18_{args.dataset}_with_masks.pth'
        save_path = os.path.join(args.save_dir, checkpoint_name)
        save_checkpoint({
            'state_dict': student.module.state_dict() if args.world_size > 1 else student.state_dict(),
            'masks': trainer.get_masks(),
            'best_acc': trainer.best_acc,
            'args': vars(args),
        }, save_path)
        print(f"✓ Saved checkpoint to {save_path}")
    
    # Phase 2: Prune (only on rank 0)
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 2: Pruning Model")
        print("="*70)
        
        pruner = ModelPruner(student.module if args.world_size > 1 else student, trainer.get_masks())
        pruned_student = pruner.prune()
        
        orig_params, pruned_params = pruner.get_params_count()
        orig_flops, pruned_flops = pruner.get_flops_count()
        params_red = (1 - pruned_params / orig_params) * 100
        flops_red = (1 - pruned_flops / orig_flops) * 100
        
        print(f"\nParameters: {orig_params:,} → {pruned_params:,} ({params_red:.2f}% reduction)")
        print(f"FLOPs: {orig_flops:,} → {pruned_flops:,} ({flops_red:.2f}% reduction)\n")
        
        # ذخیره مدل پرین‌شده برای بارگذاری در تمام رنک‌ها
        temp_pruned_path = os.path.join(args.save_dir, 'temp_pruned.pth')
        torch.save(pruned_student.state_dict(), temp_pruned_path)

    if args.world_size > 1:
        dist.barrier()

    # Load pruned model on all processes
    if rank != 0:
        temp_pruned_path = os.path.join(args.save_dir, 'temp_pruned.pth')
        pruned_student = resnet18(num_classes=NUM_CLASSES).to(device)
        pruned_student.load_state_dict(torch.load(temp_pruned_path, map_location=device))
    else:
        # رنک 0 هم مدل را از فایل بارگذاری می‌کند تا یکپارچگی حفظ شود
        temp_pruned_path = os.path.join(args.save_dir, 'temp_pruned.pth')
        pruned_student = resnet18(num_classes=NUM_CLASSES).to(device)
        pruned_student.load_state_dict(torch.load(temp_pruned_path, map_location=device))

    if args.world_size > 1:
        pruned_student = DDP(pruned_student, device_ids=[rank])
    
    # Phase 3: Fine-tune
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 3: Fine-tuning Pruned Model")
        print("="*70)
    
    optimizer = torch.optim.SGD(pruned_student.parameters(), lr=args.finetune_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    
    finetune_start_time = time.time()
    
    for epoch in range(args.finetune_epochs):
        pruned_student.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{args.finetune_epochs}", leave=False, disable=rank != 0)
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device).float()
            optimizer.zero_grad()
            outputs = pruned_student(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs > 0).long()
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        
        # Evaluation
        pruned_student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pruned_student(inputs).squeeze(1)
                preds = (outputs > 0).long()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        
        if rank == 0 and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch [{epoch+1:3d}/{args.finetune_epochs}] Train Acc: {100.*correct/total:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if rank == 0 and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': pruned_student.module.state_dict() if args.world_size > 1 else pruned_student.state_dict(),
                'accuracy': test_acc,
                'args': vars(args),
            }, os.path.join(args.save_dir, f'pruned_resnet18_{args.dataset}_best.pth'))
        
        scheduler.step()
    
    finetune_time = time.time() - finetune_start_time

    # Final Results (only on rank 0)
    if rank == 0:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Dataset: {args.dataset.upper()}")
        print(f"Teacher Accuracy: {teacher_acc:.2f}%")
        print(f"Student Best Accuracy after Distillation: {trainer.best_acc:.2f}%")
        print(f"Pruned Student Best Accuracy after Fine-tuning: {best_acc:.2f}%")
        print(f"Parameters Reduction: {params_red:.2f}%")
        print(f"FLOPs Reduction: {flops_red:.2f}%")
        print(f"Distillation Time: {distillation_time/60:.2f} minutes")
        print(f"Fine-tuning Time: {finetune_time/60:.2f} minutes")
        print(f"Total Training Time: {(distillation_time + finetune_time)/60:.2f} minutes")
        print("="*70 + "\n")
        
        summary = {
            'dataset': args.dataset,
            'teacher_acc': teacher_acc,
            'student_after_distillation': trainer.best_acc,
            'pruned_student_after_finetune': best_acc,
            'params_reduction': params_red,
            'flops_reduction': flops_red,
            'distillation_time_minutes': distillation_time/60,
            'finetune_time_minutes': finetune_time/60,
            'total_time_minutes': (distillation_time + finetune_time)/60
        }
        
        summary_path = os.path.join(args.save_dir, f'summary_{args.dataset}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✓ Summary saved to {summary_path}\n")
    
    if args.world_size > 1:
        cleanup()


def main():
    args = parse_args()
    
    # ایجاد یک فایل چک‌پوینت ساختگی برای معلم
    if not os.path.exists(args.teacher_checkpoint):
        print(f"Warning: Teacher checkpoint not found at {args.teacher_checkpoint}. Creating a dummy one.")
        os.makedirs(os.path.dirname(args.teacher_checkpoint), exist_ok=True)
        dummy_model = resnet50()
        torch.save(dummy_model.state_dict(), args.teacher_checkpoint)

    if args.world_size > 1:
        mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
    else:
        main_worker(0, args)


if __name__ == '__main__':
    main()
