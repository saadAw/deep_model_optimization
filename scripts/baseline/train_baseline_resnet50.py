#!/usr/bin/env python3
"""
ResNet50 Baseline Training Script (Modular)
===========================================
Uses the new modular utilities for training.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add the parent directory of 'deep_model_optimization' to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.logger_utils import BaseConfig  # Your new config system
from src.logger_utils import TrainingLogger # Your new logger
from src.data_utils import get_data_loaders # Your new data utils
from src.model_utils import build_model, count_model_parameters # Your new model utils
from src.training_utils import Trainer # Your new Trainer
from src.evaluation_utils import ( # Your new eval utils
    validate_model, 
    measure_inference_metrics, 
    calculate_model_size_mb_eval,
    save_final_results
)


def run_baseline_training():
    # 1. Configuration
    #    Use your BaseConfig. You can override defaults here or via argparse/YAML later.
    cfg = BaseConfig(
        data_dir=r'C:\Uni\deep_model_optimization\imagenet-mini', # Keep your path
        save_dir='./resnet50_baseline_modular_run',             # New save directory
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001, # Initial LR for SGD
        num_workers=4,
        use_pretrained=True,
        # resume_training=False # resume_training is now part of Trainer's logic via BaseConfig
    )
    # BaseConfig's __post_init__ handles Path conversion and basic validation.

    # 2. Logger Setup
    #    Use your TrainingLogger from logger_utils.py
    logger = TrainingLogger(save_dir=cfg.save_dir, log_file_name="baseline_training.log", logger_name="BaselineResNet50")
    logger.log_config(cfg) # Logs the config using the new system

    # 3. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 4. Data Loaders
    #    Use get_data_loaders from your data_utils.py
    try:
        # Note: get_data_loaders uses default ImageNet transforms.
        # If you need specific transforms from your old script, pass them as custom_transforms.
        train_loader, val_loader, num_classes = get_data_loaders(
            config=cfg # Pass the BaseConfig object
            # train_transforms_custom=your_custom_train_transforms, # If needed
            # val_transforms_custom=your_custom_val_transforms,   # If needed
        )
        logger.log_dataset_info(len(train_loader.dataset), len(val_loader.dataset), num_classes)
    except Exception as e:
        logger.error(f"Failed to setup data loaders: {e}", exc_info=True)
        sys.exit(1)

    # 5. Model Setup
    #    Use build_model from your model_utils.py
    model_arch = 'resnet50' # Specify the architecture
    try:
        model = build_model(
            model_arch=model_arch,
            num_classes=num_classes,
            pretrained=cfg.use_pretrained
            # custom_pretrained_weights_path can be used if needed
        )
        model = model.to(device)
        logger.info(f"Built model: {model_arch} (pretrained={cfg.use_pretrained})")

        param_counts = count_model_parameters(model) # Using your model_utils function
        logger.info(f"Total parameters (weights & biases): {param_counts['total_params_wb']:,}")
        logger.info(f"Trainable parameters (weights & biases): {param_counts['non_zero_params_wb']:,}") # Assumes all are trainable initially

    except Exception as e:
        logger.error(f"Failed to build model: {e}", exc_info=True)
        sys.exit(1)

    # 6. Optimizer, Criterion, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=0.9,
        weight_decay=1e-4 # From your old script
    )
    # Optional: Add an LR scheduler if needed by the Trainer
    # Example: scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = None # Or your preferred scheduler

    # 7. Trainer Initialization
    #    Use your Trainer class from training_utils.py
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg, # Pass the BaseConfig object
        logger=logger,
        lr_scheduler=scheduler
    )

    # 8. Start Training
    training_start_time = time.time()
    try:
        trainer.train(train_loader, val_loader) # This handles epochs, checkpointing, etc.
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        # Trainer might have saved a checkpoint already.
    except Exception as e:
        logger.error(f"Training failed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)
    
    training_duration_seconds = time.time() - training_start_time

    # 9. Final Evaluation (using your evaluation_utils.py)
    logger.info("=== Final Model Evaluation (using best checkpoint) ===")
    
    # Load the best model saved by the Trainer
    best_model_path = cfg.save_dir / 'best_model.pth'
    if best_model_path.exists():
        # Re-build the model structure (or use the current `model` instance and load state_dict)
        # For safety, re-build and load, especially if further operations modified `model`
        final_eval_model = build_model(
            model_arch=model_arch, 
            num_classes=num_classes, 
            pretrained=False # Structure only
        )
        final_eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_eval_model = final_eval_model.to(device)
        final_eval_model.eval()
        logger.info(f"Loaded best model from {best_model_path} for final evaluation.")

        best_val_loss, best_val_acc = validate_model(
            model=final_eval_model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger
        )
        logger.info(f"Re-validated Best Model: Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")

        model_size_mb = calculate_model_size_mb_eval(final_eval_model) # From evaluation_utils
        logger.info(f"Best Model Size: {model_size_mb:.2f} MB")

        inference_metrics = measure_inference_metrics(
            model=final_eval_model,
            val_loader=val_loader, # Ensure val_loader is suitable for timing
            device=device,
            logger=logger
        )
        logger.info(f"Inference Metrics: {inference_metrics}")

        # Compile and save final metrics
        final_results_data = {
            'experiment_name': 'resnet50_baseline_modular',
            'model_architecture': model_arch,
            'best_validation_accuracy': trainer.best_val_acc, # From trainer history
            'final_reval_accuracy_on_best_model': best_val_acc,
            'model_size_mb': model_size_mb,
            **inference_metrics,
            'total_training_time_seconds': training_duration_seconds,
            'config_used': cfg.to_dict() # Save the config again for clarity
        }
        save_final_results(final_results_data, cfg.save_dir, filename="summary_metrics.json")

    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}. Skipping final evaluation details.")

    logger.info("Baseline training script finished.")


if __name__ == '__main__':
    run_baseline_training()