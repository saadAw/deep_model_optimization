import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # For type checking and scheduler.step(loss)

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union # Added Tuple, Union

# Project-specific imports
from .config import BaseConfig
from .logger_utils import TrainingLogger


class Trainer:
    """
    A general-purpose class for training PyTorch models.
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 config: BaseConfig,
                 logger: TrainingLogger,
                 lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (optim.Optimizer): The optimizer for training.
            criterion (nn.Module): The loss function.
            device (torch.device): The device to train on (e.g., 'cuda', 'cpu').
            config (BaseConfig): Configuration object with training parameters.
            logger (TrainingLogger): Logger for recording training progress.
            lr_scheduler (Optional[optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger
        self.lr_scheduler = lr_scheduler

        self.history: Dict[str, list] = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': [], 'lr_values': [] # Added lr_values to history
        }
        self.best_val_acc: float = 0.0
        self.start_epoch: int = 0

        self.model.to(self.device) # Ensure model is on the correct device

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Performs one epoch of training.

        Args:
            train_loader (DataLoader): DataLoader for the training set.

        Returns:
            Tuple[float, float]: Average training loss and accuracy for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0 and batch_idx > 0: # Log every 100 batches
                self.logger.info(
                    f"Epoch {self.start_epoch + len(self.history['train_loss']) + 1} "
                    f"Batch {batch_idx}/{len(train_loader)}: Train Loss: {loss.item():.4f}"
                )
        
        epoch_loss = total_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Performs one epoch of validation.

        Args:
            val_loader (DataLoader): DataLoader for the validation set.

        Returns:
            Tuple[float, float]: Average validation loss and accuracy for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = total_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """
        Saves a training checkpoint.

        Args:
            epoch (int): The current epoch number.
            is_best (bool): True if this checkpoint has the best validation accuracy so far.
        """
        checkpoint_dir = Path(self.config.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config.to_dict() # Save BaseConfig as dict
        }
        if self.lr_scheduler:
            checkpoint_data['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        latest_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint_data, latest_checkpoint_path)
        self.logger.info(f"Saved latest checkpoint to {latest_checkpoint_path}")

        if is_best:
            best_model_path = checkpoint_dir / 'best_model.pth'
            # For best_model.pth, typically only model state_dict is saved, but saving more is also fine.
            # Let's save the full checkpoint data for consistency here, or just model state:
            torch.save(self.model.state_dict(), best_model_path) 
            self.logger.info(f"Saved best model to {best_model_path} (Val Acc: {self.best_val_acc:.4f})")

        history_path = checkpoint_dir / 'training_history.json'
        try:
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            self.logger.info(f"Saved training history to {history_path}")
        except IOError as e:
            self.logger.error(f"Failed to save training history: {e}")


    def _load_checkpoint(self):
        """
        Loads a training checkpoint if resume_training is True in config and checkpoint exists.
        Updates model, optimizer, scheduler (if any), history, start_epoch, and best_val_acc.
        """
        if not self.config.evaluate_only and hasattr(self.config, 'resume_training') and self.config.resume_training:
            checkpoint_path = Path(self.config.save_dir) / 'latest_checkpoint.pth'
            if checkpoint_path.exists():
                self.logger.info(f"Attempting to load checkpoint from {checkpoint_path}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)

                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                        self.logger.info("Loaded LR scheduler state.")

                    self.start_epoch = checkpoint['epoch'] + 1
                    self.best_val_acc = checkpoint.get('best_val_acc', 0.0) # Default to 0.0 if not in older checkpoints
                    self.history = checkpoint.get('history', self.history) # Default to current history if not found

                    self.logger.info(f"Resumed training from epoch {self.start_epoch}. Best Val Acc: {self.best_val_acc:.4f}")
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"Optimizer LR resumed to: {current_lr:.6f}")

                except Exception as e:
                    self.logger.error(f"Failed to load checkpoint: {e}. Starting training from scratch.")
                    self.start_epoch = 0
                    self.best_val_acc = 0.0
                    # Optionally clear history if checkpoint load fails catastrophically
                    # self.history = {k: [] for k in self.history} 
            else:
                self.logger.info("Resume training requested, but no checkpoint found. Starting from scratch.")
        else:
            if self.config.evaluate_only:
                 self.logger.info("Evaluate only mode: Not loading optimizer/scheduler states from checkpoint.")
                 # In eval mode, we might still want to load model weights from a specific path (handled outside this method)
            elif not hasattr(self.config, 'resume_training') or not self.config.resume_training:
                 self.logger.info("Not resuming training (resume_training is False or not set). Starting from scratch.")


    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        The main training loop.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
        """
        self.logger.info("=== Starting Training ===")
        self._load_checkpoint() # Load checkpoint if resuming

        total_training_start_time = time.time()

        epochs_to_run = self.config.num_epochs - self.start_epoch
        if epochs_to_run <= 0 and not self.config.evaluate_only:
            self.logger.info(f"Training already completed up to epoch {self.config.num_epochs}. "
                             f"Current start epoch is {self.start_epoch}.")
            # Optionally, run final evaluation here if needed, or just exit
            final_val_loss, final_val_acc = self.validate_epoch(val_loader)
            self.logger.info(f"Final validation after loading checkpoint: Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.4f}")
            return

        if not self.config.evaluate_only:
            self.logger.info(f"Training for {epochs_to_run} epochs (from epoch {self.start_epoch + 1} to {self.config.num_epochs}).")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"--- Epoch {epoch + 1}/{self.config.num_epochs} --- Learning Rate: {current_lr:.7f} ---")

            if not self.config.evaluate_only:
                train_loss, train_acc = self.train_epoch(train_loader)
            else: # In evaluate_only mode, skip training epoch
                train_loss, train_acc = 0.0, 0.0
                self.logger.info("evaluate_only is True, skipping training epoch.")

            val_loss, val_acc = self.validate_epoch(val_loader)

            # Record metrics
            epoch_time_taken = time.time() - epoch_start_time
            if not self.config.evaluate_only: # Only append history if actually training
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['epoch_times'].append(epoch_time_taken)
                self.history['lr_values'].append(current_lr)

            self.logger.info(
                f"Epoch {epoch + 1} Summary: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time_taken:.2f}s"
            )

            is_best = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                is_best = True
                self.logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}")

            if not self.config.evaluate_only: # Only save checkpoint if training
                self._save_checkpoint(epoch=epoch, is_best=is_best)

            # LR Scheduler step
            if self.lr_scheduler and not self.config.evaluate_only:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss) # ReduceLROnPlateau needs a metric
                else:
                    self.lr_scheduler.step() # For other schedulers like StepLR, MultiStepLR, CosineAnnealingLR

            if self.config.evaluate_only:
                self.logger.info("evaluate_only is True. Evaluation for one epoch completed.")
                break # Exit loop after one validation epoch if in evaluate_only mode


        if not self.config.evaluate_only:
            total_training_time = time.time() - total_training_start_time
            self.logger.info(f"=== Training Completed in {total_training_time:.2f} seconds ===")
            self.logger.info(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        else:
            self.logger.info("=== Evaluation Completed ===")


# Example Usage (Illustrative - requires more setup to run directly)
if __name__ == '__main__':
    # This is a conceptual example.
    # To run this, you'd need to:
    # 1. Define a dummy BaseConfig, model, optimizer, criterion, device, logger.
    # 2. Create dummy DataLoaders.

    print("Illustrative example of Trainer class usage.")

    # --- Dummy Components ---
    from .model_utils import build_model # Assuming model_utils.py is in the same directory
    
    # 1. Config
    class DummyConfig(BaseConfig):
        resume_training: bool = False # Add resume_training for testing _load_checkpoint
        # evaluate_only is part of BaseConfig by default

    dummy_cfg = DummyConfig(
        save_dir='./temp_trainer_run', 
        num_epochs=3, 
        batch_size=2,
        learning_rate=0.01,
        # evaluate_only=False # Test training
    )
    Path(dummy_cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # 2. Logger
    dummy_logger = TrainingLogger(save_dir=Path(dummy_cfg.save_dir), log_file_name="trainer_test.log")

    # 3. Device
    dummy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_logger.info(f"Using device: {dummy_device}")

    # 4. Model
    # Assuming num_classes is known for the dummy data
    num_dummy_classes = 2 
    dummy_model = build_model(model_arch='resnet18', num_classes=num_dummy_classes, pretrained=False)
    dummy_model = dummy_model.to(dummy_device)

    # 5. Optimizer, Criterion, Scheduler
    dummy_optimizer = optim.SGD(dummy_model.parameters(), lr=dummy_cfg.learning_rate)
    dummy_criterion = nn.CrossEntropyLoss()
    dummy_scheduler = optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1, gamma=0.1) # Step every epoch


    # 6. DataLoaders (very basic dummy)
    # Create dummy data and datasets
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=10, num_classes=2):
            self.num_samples = num_samples
            # Generate random data: (batch, channels, height, width)
            self.data = torch.randn(num_samples, 3, 32, 32) 
            self.targets = torch.randint(0, num_classes, (num_samples,))
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    dummy_train_dataset = DummyDataset(num_samples=20, num_classes=num_dummy_classes)
    dummy_val_dataset = DummyDataset(num_samples=10, num_classes=num_dummy_classes)

    dummy_train_loader = DataLoader(dummy_train_dataset, batch_size=dummy_cfg.batch_size)
    dummy_val_loader = DataLoader(dummy_val_dataset, batch_size=dummy_cfg.batch_size)
    
    dummy_logger.info(f"Dummy TrainLoader: {len(dummy_train_loader)} batches, {len(dummy_train_dataset)} samples")
    dummy_logger.info(f"Dummy ValLoader: {len(dummy_val_loader)} batches, {len(dummy_val_dataset)} samples")


    # --- Instantiate Trainer ---
    trainer = Trainer(
        model=dummy_model,
        optimizer=dummy_optimizer,
        criterion=dummy_criterion,
        device=dummy_device,
        config=dummy_cfg,
        logger=dummy_logger,
        lr_scheduler=dummy_scheduler
    )
    
    # --- Test _load_checkpoint (Optional: create a dummy checkpoint first) ---
    # trainer._load_checkpoint() # Test loading if checkpoint exists

    # --- Start Training ---
    try:
        trainer.train(dummy_train_loader, dummy_val_loader)
        dummy_logger.info("Example training run completed.")

        # --- Test evaluate_only mode (Optional) ---
        # dummy_cfg_eval = DummyConfig(save_dir='./temp_trainer_run', num_epochs=1, evaluate_only=True)
        # trainer_eval = Trainer(model=dummy_model, optimizer=dummy_optimizer, criterion=dummy_criterion,
        #                        device=dummy_device, config=dummy_cfg_eval, logger=dummy_logger)
        # # We would typically load a specific model checkpoint for evaluation.
        # # For this test, it will just run validate_epoch on the current model state.
        # if (Path(dummy_cfg.save_dir) / 'best_model.pth').exists():
        #     state_dict = torch.load(Path(dummy_cfg.save_dir) / 'best_model.pth', map_location=dummy_device)
        #     trainer_eval.model.load_state_dict(state_dict)
        #     dummy_logger.info("Loaded best model for evaluation test.")
        # trainer_eval.train(dummy_train_loader, dummy_val_loader) # Runs only validation
        # dummy_logger.info("Example evaluation run completed.")


    except Exception as e:
        dummy_logger.error(f"Error during example Trainer usage: {e}", exc_info=True)
    finally:
        # Clean up dummy save directory
        import shutil
        # temp_dir_to_remove = Path(dummy_cfg.save_dir)
        # if temp_dir_to_remove.exists():
        #     print(f"Note: To clean up, remove the directory: {temp_dir_to_remove.resolve()}")
            # shutil.rmtree(temp_dir_to_remove) # Uncomment to automatically clean up
        pass
    
    print(f"Example finished. Check logs in {Path(dummy_cfg.save_dir).resolve()}")
