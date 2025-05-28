#!/usr/bin/env python3
"""
ResNet50 Baseline Training Script for ImageNet-Mini
==================================================

A clean, robust training script for establishing baseline metrics.
Supports resuming from checkpoints and comprehensive metric logging.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights

# Removed unused imports: import io, import types


@dataclass
class Config:
    """Training configuration with validation."""
    # Paths
    data_dir: str = r'C:\Uni\deep_model_optimization\imagenet-mini'
    save_dir: str = './resnet50_baseline_e30_run' # Example save directory name
    
    # Training parameters  
    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    num_workers: int = 4
    
    # Model settings
    use_pretrained: bool = True
    resume_training: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_workers < 0:
             raise ValueError("num_workers cannot be negative")
        
        # Convert to Path objects for easier handling
        self.data_dir = Path(self.data_dir)
        self.save_dir = Path(self.save_dir)


class TrainingLogger:
    """Handles logging setup and training metrics."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_config(self, config: Config):
        """Log and save training configuration."""
        self.logger.info("=== Training Configuration ===")
        # Convert Path objects to strings and then use asdict
        config_dict = asdict(config)
        config_dict['data_dir'] = str(config_dict['data_dir'])
        config_dict['save_dir'] = str(config_dict['save_dir'])
        
        for key, value in config_dict.items():
            self.logger.info(f"{key}: {value}")
        
        # Save config to file
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def log_dataset_info(self, train_size: int, val_size: int, num_classes: int):
        """Log dataset information."""
        self.logger.info("=== Dataset Information ===")
        self.logger.info(f"Number of classes: {num_classes}")
        self.logger.info(f"Training samples: {train_size:,}")
        self.logger.info(f"Validation samples: {val_size:,}")


class ResNet50Trainer:
    """Main training class for ResNet50 baseline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger_handler = TrainingLogger(config.save_dir)
        self.logger = self.logger_handler.logger
        
        # Training state
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 
            'epoch_times': []
        }
        self.best_val_acc = 0.0
        self.start_epoch = 0
        
        self.logger.info(f"Using device: {self.device}")
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader, int]:
        """Setup data loaders with proper transforms."""
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Load datasets
        try:
            train_dataset = datasets.ImageFolder(
                self.config.data_dir / 'train', 
                train_transforms
            )
            val_dataset = datasets.ImageFolder(
                self.config.data_dir / 'val', 
                val_transforms
            )
        except FileNotFoundError as e:
            self.logger.error(f"Dataset not found: {e}")
            self.logger.error(f"Ensure {self.config.data_dir} contains 'train' and 'val' folders")
            raise
        
        num_classes = len(train_dataset.classes)
        
        # Log dataset info
        self.logger_handler.log_dataset_info(
            len(train_dataset), len(val_dataset), num_classes
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True, 
            num_workers=self.config.num_workers,
            pin_memory=True,
            # persistent_workers=self.config.num_workers > 0 # Removed for broader compatibility/simplicity unless needed
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            # persistent_workers=self.config.num_workers > 0 # Removed
        )
        
        return train_loader, val_loader, num_classes
    
    def setup_model(self, num_classes: int):
        """Initialize model and optimizer."""
        self.logger.info(f"Setting up ResNet50 (pretrained={self.config.use_pretrained})")
        
        weights = ResNet50_Weights.IMAGENET1K_V1 if self.config.use_pretrained else None
        self.model = models.resnet50(weights=weights)
        
        # Modify classifier for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=1e-4  # Added weight decay for better generalization
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if resuming training."""
        checkpoint_path = self.config.save_dir / 'latest_checkpoint.pth'
        
        if not self.config.resume_training:
            return False
            
        if not checkpoint_path.exists():
            self.logger.warning(f"Resume requested but no checkpoint found at {checkpoint_path}")
            return False
        
        try:
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.history = checkpoint['history']
            
            self.logger.info(f"Resumed from epoch {self.start_epoch}")
            self.logger.info(f"Best validation accuracy so far: {self.best_val_acc:.4f}")
            # Note: Optimizer LR resumed from checkpoint
            self.logger.info(f"Optimizer LR resumed as {self.optimizer.param_groups[0]['lr']:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            self.logger.info("Starting training from scratch")
            # Reset state if loading fails
            self.start_epoch = 0 
            self.best_val_acc = 0.0
            self.history = {
                'train_loss': [], 'train_acc': [], 
                'val_loss': [], 'val_acc': [], 
                'epoch_times': []
            }
            return False
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        # Convert config to dict, ensuring Path objects are strings
        config_dict = asdict(self.config)
        config_dict['data_dir'] = str(config_dict['data_dir'])
        config_dict['save_dir'] = str(config_dict['save_dir'])

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc, # Save best_val_acc found *so far*
            'history': self.history, # Save updated history
            'config': config_dict # Save the run configuration
        }
        
        checkpoint_path = self.config.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info("Saved latest checkpoint.")

        # Save history separately for easy access
        history_path = self.config.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2) # Use indent=2 for smaller file size
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for a progress bar if installed (optional)
        # try:
        #     from tqdm import tqdm
        #     train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        # except ImportError:
        train_iterator = train_loader # Fallback if tqdm is not installed

        for batch_idx, (inputs, targets) in enumerate(train_iterator):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log progress less frequently if using tqdm
            # if (batch_idx + 1) % 100 == 0:
            #     self.logger.info(
            #         f"Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
            #         f"Train Loss: {loss.item():.4f}"
            #     )
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for a progress bar if installed (optional)
        # try:
        #     from tqdm import tqdm
        #     val_iterator = tqdm(val_loader, desc=f"Validation")
        # except ImportError:
        val_iterator = val_loader # Fallback

        with torch.no_grad():
            for inputs, targets in val_iterator:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def measure_inference_speed(self, val_loader: DataLoader) -> Dict[str, float]:
        """Measure model inference speed."""
        self.logger.info("Measuring inference speed...")
        self.model.eval()
        
        # Create a dedicated loader for consistent timing
        # Use a separate DataLoader instance or rewind if possible, 
        # but creating a new one is simplest here.
        timing_loader = DataLoader(
            val_loader.dataset, # Use the same dataset
            batch_size=val_loader.batch_size,
            shuffle=False, # Must not shuffle for consistency
            num_workers=0, # Often better to time on main process to avoid worker overhead noise
            pin_memory=True
        )

        # Warmup
        warmup_batches = 5
        # Ensure we have enough batches to warm up and time
        if len(timing_loader) < warmup_batches + 1: 
            self.logger.warning(f"Not enough batches ({len(timing_loader)}) in validation loader for sufficient warmup and timing.")
            return {
                'images_per_second': 0, 
                'latency_ms_per_image': 0, 
                'total_images_measured': 0, 
                'total_time_seconds': 0
            }

        with torch.no_grad():
            for i, (inputs, _) in enumerate(timing_loader):
                if i >= warmup_batches:
                    break
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # Ensure GPU operations are finished

        # Timing
        timing_batches = min(50, len(timing_loader) - warmup_batches) # Time over remaining batches
        total_images = 0
        
        start_time = time.time()
        # Reset iterator after warmup
        timing_loader_iter = iter(timing_loader)
        # Skip warmup batches again (less efficient than just iterating, but safer)
        # Simpler: just iterate from the start and time after warmup batches
        start_timing_batch_idx = warmup_batches
        
        with torch.no_grad():
             for batch_idx, (inputs, _) in enumerate(timing_loader):
                 if batch_idx < start_timing_batch_idx:
                     continue # Skip warmup batches
                 if batch_idx >= start_timing_batch_idx + timing_batches:
                     break # Stop after timing batches
                     
                 inputs = inputs.to(self.device)
                 _ = self.model(inputs)
                 total_images += inputs.size(0)
                 
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # Ensure GPU operations are finished
        
        inference_time = time.time() - start_time
        
        if total_images == 0 or inference_time <= 0: # Handle edge cases
             self.logger.warning("Inference timing yielded zero images or zero time.")
             images_per_sec = 0
             latency_ms = 0
        else:
             images_per_sec = total_images / inference_time
             latency_ms = (inference_time / total_images) * 1000
        
        return {
            'images_per_second': float(images_per_sec), # Ensure float for JSON
            'latency_ms_per_image': float(latency_ms),   # Ensure float for JSON
            'total_images_measured': total_images,
            'total_time_seconds': float(inference_time)  # Ensure float for JSON
        }
    
    def calculate_model_size(self) -> float:
        """Calculate model size in MB."""
        # Save model state dict to a temporary buffer to get its size
        buffer = io.BytesIO()
        # Save state_dict to buffer, map_location='cpu' might be safer/more standard for size calc
        torch.save(self.model.state_dict(), buffer, map_location='cpu') 
        model_size_bytes = buffer.getbuffer().nbytes
        return model_size_bytes / (1024 ** 2)  # Convert to MB
    
    def train(self):
        """Main training loop."""
        # Setup
        self.logger_handler.log_config(self.config)
        try:
            train_loader, val_loader, num_classes = self.setup_data_loaders()
        except Exception as e:
             self.logger.error(f"Failed to setup data loaders: {e}")
             sys.exit(1) # Exit if data loading fails

        self.setup_model(num_classes)
        
        # Load checkpoint AFTER model is set up
        self.load_checkpoint() 
        
        self.logger.info("=== Starting Training ===")
        total_start_time = time.time()
        
        # Ensure epochs to run is correct if resuming
        epochs_to_run = self.config.num_epochs - self.start_epoch
        if epochs_to_run <= 0:
            self.logger.info(f"Training already completed (Current epoch: {self.start_epoch}, Target epochs: {self.config.num_epochs}). Skipping training loop.")
        else:
            self.logger.info(f"Running for {epochs_to_run} epochs starting from epoch {self.start_epoch + 1}")

            for epoch in range(self.start_epoch, self.config.num_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_loss, train_acc = self.train_epoch(train_loader, epoch)
                
                # Validation phase
                val_loss, val_acc = self.validate(val_loader)
                
                # Record metrics
                epoch_time = time.time() - epoch_start_time
                self.history['train_loss'].append(float(train_loss))
                self.history['train_acc'].append(float(train_acc))
                self.history['val_loss'].append(float(val_loss))
                self.history['val_acc'].append(float(val_acc))
                self.history['epoch_times'].append(float(epoch_time))
                
                # Logging
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                    f"Time: {epoch_time:.1f}s"
                )
                
                # Save checkpoint
                self.save_checkpoint(epoch)
                
                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    best_model_path = self.config.save_dir / 'best_model.pth'
                    torch.save(self.model.state_dict(), best_model_path)
                    self.logger.info(f"New best model saved! Val Acc: {val_acc:.4f}")
                
                # Optional: Adjust learning rate (e.g., StepLR, CosineAnnealingLR)
                # if (epoch + 1) % 10 == 0:
                #     for param_group in self.optimizer.param_groups:
                #         param_group['lr'] *= 0.1
                #     self.logger.info(f"Learning rate reduced to {self.optimizer.param_groups[0]['lr']:.6f}")


        total_time = time.time() - total_start_time
        self.logger.info(f"Training completed in {total_time:.1f} seconds")
        
        # Final evaluation
        self.evaluate_final_model(val_loader, total_time)
    
    def evaluate_final_model(self, val_loader: DataLoader, training_time: float):
        """Evaluate the final model and save comprehensive metrics."""
        self.logger.info("=== Final Model Evaluation ===")
        
        # Load best model state dict for analysis (ensure it's available)
        best_model_path = self.config.save_dir / 'best_model.pth'
        if not best_model_path.exists():
            self.logger.error(f"Best model checkpoint not found at {best_model_path}. Skipping final evaluation.")
            return

        # Re-instantiate model structure to load state dict
        try:
            # Get number of classes from the dataset object
            num_classes = len(val_loader.dataset.classes) 
            temp_model = models.resnet50(weights=None) 
            num_ftrs = temp_model.fc.in_features
            temp_model.fc = nn.Linear(num_ftrs, num_classes)

            # Load the state dict from the best model path (map_location='cpu' is robust)
            temp_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
            temp_model = temp_model.to(self.device) # Move to device for inference
            self.logger.info(f"Successfully loaded best model from {best_model_path}")

        except Exception as e:
            self.logger.error(f"Error loading best model for final evaluation: {e}")
            return

        # Calculate metrics using the temporary loaded model
        model_size_mb = self.calculate_model_size() # Using self.model here which was potentially modified during training
        # Better to use the loaded temp_model for size calculation:
        buffer = io.BytesIO()
        torch.save(temp_model.state_dict(), buffer, map_location='cpu')
        model_size_mb_loaded = buffer.getbuffer().nbytes / (1024 ** 2)
        self.logger.info(f"Model size (loaded best checkpoint): {model_size_mb_loaded:.2f} MB")


        inference_metrics = self.measure_inference_speed(val_loader)
        
        # Compile final metrics
        final_metrics = {
            'training_completed': True,
            'epochs_trained': self.config.num_epochs, # Total epochs planned
            'epochs_actually_run': self.config.num_epochs - self.start_epoch, # Epochs completed in this run
            'starting_epoch': self.start_epoch + 1, # Which epoch the run started from
            'best_validation_accuracy': float(self.best_val_acc),
            'total_training_time_seconds': float(training_time),
            'model_size_mb_loaded': float(model_size_mb_loaded), # Use the size of the loaded model
            **inference_metrics,
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.config).items()}
        }
        
        # Save metrics
        metrics_path = self.config.save_dir / 'final_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Log summary
        self.logger.info("--- Final Summary ---")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        self.logger.info(f"Model size (best checkpoint): {model_size_mb_loaded:.2f} MB")
        self.logger.info(f"Inference speed: {inference_metrics['images_per_second']:.1f} images/sec")
        self.logger.info(f"Inference latency: {inference_metrics['latency_ms_per_image']:.2f} ms/image")
        self.logger.info(f"All results saved to: {self.config.save_dir}")


def main():
    """Main entry point."""
    try:
        config = Config()
        trainer = ResNet50Trainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user") # Print directly to console
        logging.info("Training interrupted by user") # Log the interruption
    except Exception as e:
        logging.exception("Training failed with an unexpected error") # Log the full traceback
        # print(f"Training failed with error: {e}") # Print summary to console
        sys.exit(1) # Exit with a non-zero status code to indicate failure


if __name__ == '__main__':
    # Add io import here since it's used in calculate_model_size and evaluate_final_model
    # If used only within methods, importing inside the method is also an option
    import io 
    main()