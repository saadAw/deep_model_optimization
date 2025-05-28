#!/usr/bin/env python3
"""
Utility functions for Deep Model Optimization Project.
====================================================

Contains reusable functions for data loading, model setup,
evaluation, metric calculation, logging, saving, and loading.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

import io
import time
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """Base configuration for training and optimization runs."""
    # Paths
    data_dir: str = r'C:\Uni\deep_model_optimization\imagenet-mini'
    save_dir: str = './experiment_run'

    # Training parameters
    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    num_workers: int = 4

    # Model settings
    use_pretrained: bool = True
    evaluate_only: bool = False

    def __post_init__(self):
        """Validate configuration and convert paths."""
        if self.num_epochs < 0:
            raise ValueError("num_epochs must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate < 0:
            raise ValueError("learning_rate cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        # Convert to Path objects
        self.data_dir = Path(self.data_dir)
        self.save_dir = Path(self.save_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to dictionary with string paths."""
        data = asdict(self)
        data['data_dir'] = str(data['data_dir'])
        data['save_dir'] = str(data['save_dir'])
        return data


class TrainingLogger:
    """Handles logging setup and configuration logging."""

    def __init__(self, save_dir: Path, log_file_name: str = 'run.log'):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.save_dir / log_file_name

        # Setup logger with unique name to avoid conflicts
        logger_name = f"training_logger_{id(self)}"
        self.logger = logging.getLogger(logger_name)
        
        # Only configure if not already configured
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # File handler
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log_config(self, config: Config):
        """Log and save run configuration."""
        self.logger.info("=== Configuration ===")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            self.logger.info(f"{key}: {value}")

        # Save config to file
        config_path = self.save_dir / 'config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.logger.info(f"Config saved to {config_path}")
        except IOError as e:
            self.logger.error(f"Failed to save config file: {e}")

    def log_dataset_info(self, train_size: int, val_size: int, num_classes: int):
        """Log dataset information."""
        self.logger.info("=== Dataset Information ===")
        self.logger.info(f"Number of classes: {num_classes}")
        self.logger.info(f"Training samples: {train_size:,}")
        self.logger.info(f"Validation samples: {val_size:,}")


def setup_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, int]:
    """Setup data loaders with proper transforms."""
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

    try:
        train_dataset = datasets.ImageFolder(config.data_dir / 'train', train_transforms)
        val_dataset = datasets.ImageFolder(config.data_dir / 'val', val_transforms)
    except FileNotFoundError as e:
        print(f"ERROR: Dataset not found: {e}", file=sys.stderr)
        print(f"Ensure {config.data_dir} contains 'train' and 'val' folders", file=sys.stderr)
        raise

    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )

    return train_loader, val_loader, num_classes


def setup_model_structure(num_classes: int) -> nn.Module:
    """Sets up the ResNet50 model structure without loading weights."""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def validate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Validate the model on a data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def measure_inference_speed(model: nn.Module, data_loader: DataLoader, device: torch.device, 
                          num_warmup_batches: int = 5, num_timing_batches: int = 50) -> Dict[str, float]:
    """Measure model inference speed on the specified device."""
    print("Measuring inference speed...")
    model.eval()

    try:
        timing_loader = DataLoader(
            data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    except Exception as e:
        print(f"WARNING: Failed to create timing data loader: {e}")
        return {
            'images_per_second': 0.0,
            'latency_ms_per_image': 0.0,
            'total_images_measured': 0,
            'total_time_seconds': 0.0
        }

    total_batches_available = len(timing_loader)
    if total_batches_available < num_warmup_batches + num_timing_batches:
        num_timing_batches = max(0, total_batches_available - num_warmup_batches)
        if num_timing_batches == 0:
            return {
                'images_per_second': 0.0,
                'latency_ms_per_image': 0.0,
                'total_images_measured': 0,
                'total_time_seconds': 0.0
            }

    with torch.no_grad():
        # Warmup
        timing_iterator = iter(timing_loader)
        for _ in range(num_warmup_batches):
            try:
                inputs, _ = next(timing_iterator)
                inputs = inputs.to(device)
                _ = model(inputs)
            except StopIteration:
                break

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Timing
        total_images = 0
        start_time = time.time()

        for _ in range(num_timing_batches):
            try:
                inputs, _ = next(timing_iterator)
                inputs = inputs.to(device)
                _ = model(inputs)
                total_images += inputs.size(0)
            except StopIteration:
                break

        if device.type == 'cuda':
            torch.cuda.synchronize()

        inference_time = time.time() - start_time

        if total_images == 0 or inference_time <= 0:
            images_per_sec = 0.0
            latency_ms = 0.0
        else:
            images_per_sec = total_images / inference_time
            latency_ms = (inference_time / total_images) * 1000.0

        return {
            'images_per_second': float(images_per_sec),
            'latency_ms_per_image': float(latency_ms),
            'total_images_measured': total_images,
            'total_time_seconds': float(inference_time)
        }


def calculate_model_size(state_dict: torch.Tensor) -> float:
    """Calculate size of model state dict in MB."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    model_size_bytes = buffer.getbuffer().nbytes
    return model_size_bytes / (1024 ** 2)


def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total and non-zero parameters in the model."""
    total_params = 0
    non_zero_params = 0
    
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            total_params += numel
            if numel > 0:
                non_zero_params += torch.count_nonzero(param).item()

    return {
        'total_params': total_params,
        'non_zero_params': non_zero_params
    }


def save_pruning_metrics(save_path: Path, metrics: Dict[str, Any]):
    """Saves metrics dictionary to a JSON file."""
    try:
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {save_path}")
    except IOError as e:
        print(f"ERROR: Failed to save metrics file {save_path}: {e}")


def load_model_state_dict(model_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """Loads a model state dictionary from a file."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model state dict not found at {model_path}")
    
    try:
        state_dict = torch.load(str(model_path), map_location=device)
        print(f"State dict loaded from {model_path} to {device}.")
        return state_dict
    except Exception as e:
        print(f"ERROR: Failed to load state dict from {model_path}: {e}")
        raise