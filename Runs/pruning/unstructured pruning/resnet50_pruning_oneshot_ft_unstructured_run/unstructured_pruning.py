#!/usr/bin/env python3
"""
ResNet50 Unstructured Pruning Script
===================================

Applies unstructured magnitude pruning and fine-tuning to a baseline model.
Tests different sparsity rates with robust checkpointing and error handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field

try:
    from utils import (
        Config as BaseConfig,
        TrainingLogger,
        setup_data_loaders,
        setup_model_structure,
        validate_model,
        measure_inference_speed,
        calculate_model_size,
        count_model_parameters,
        save_pruning_metrics,
        load_model_state_dict,
        convert_to_sparse,
        set_seed
    )
except ImportError:
    print("Error: Could not import from utils.py. Make sure it's in the same directory or PYTHONPATH.")
    sys.exit(1)


@dataclass
class PruningConfig(BaseConfig):
    """Configuration for unstructured pruning runs."""
    # Override base settings
    baseline_model_path: str = './resnet50_baseline_e30_run/best_model.pth'
    save_dir: str = './resnet50_pruning_unstructured_run'
    
    # Pruning specific settings
    pruning_method: str = 'global_unstructured_l1'
    sparsity_rates: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    
    # Fine-tuning parameters
    ft_epochs: int = 15
    ft_learning_rate: float = 0.00005
    ft_momentum: float = 0.9
    ft_weight_decay: float = 1e-4
    
    # Resume settings
    resume_pruning: bool = False
    skip_completed: bool = True  # Skip sparsity levels that already have results

    use_sparse_storage: bool = True  # Whether to convert to sparse tensors

    def __post_init__(self):
        super().__post_init__()
        
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if not Path(self.baseline_model_path).exists():
            raise FileNotFoundError(f"Baseline model not found: {self.baseline_model_path}")
        
        if not all(0.0 <= r <= 1.0 for r in self.sparsity_rates):
            raise ValueError("Sparsity rates must be between 0.0 and 1.0")
        
        if self.ft_epochs < 0:
            raise ValueError("ft_epochs must be non-negative")
        if self.ft_learning_rate <= 0:
            raise ValueError("ft_learning_rate must be positive")


class PruningExperiment:
    """Handles the complete pruning experiment workflow."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger_handler = TrainingLogger(config.save_dir, 'pruning_experiment.log')
        self.logger = self.logger_handler.logger
        
        # State tracking
        self.baseline_results = None
        self.completed_sparsities = set()
        
        self.logger.info(f"Using device: {self.device}")
    
    def load_completed_experiments(self):
        """Check which sparsity levels are already completed."""
        if not self.config.skip_completed:
            return
            
        for sparsity in self.config.sparsity_rates:
            results_file = self.config.save_dir / f'metrics_{sparsity*100:.0f}_unstructured.json'
            if results_file.exists():
                self.completed_sparsities.add(sparsity)
                self.logger.info(f"Found existing results for sparsity {sparsity*100:.1f}%")
    
    def apply_global_unstructured_pruning(self, model: nn.Module, sparsity_rate: float):
        """Apply global unstructured L1 pruning to Conv2d and Linear layers."""
        self.logger.info(f"Applying global unstructured pruning to {sparsity_rate*100:.1f}% sparsity...")
        
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                    parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            self.logger.warning("No prunable parameters found!")
            return
        
        # Avoid 100% sparsity exactly, as it might lead to issues.
        # Pruning utilities should ideally handle it, but this is a safe guard.
        amount_to_prune = min(sparsity_rate, 1.0 - 1e-6 if sparsity_rate == 1.0 else sparsity_rate)
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount_to_prune,
        )
        self.logger.info(f"Pruning applied to {len(parameters_to_prune)} parameter tensors.")
    
    def remove_pruning_masks(self, model: nn.Module):
        """Remove pruning reparameterization from the model, making pruning permanent."""
        self.logger.info("Removing pruning masks...")
        pruned_count = 0
        
        for name, module in model.named_modules():
            # Check if module has weight parameter and if it's pruned (has a weight_mask)
            if hasattr(module, 'weight') and prune.is_pruned(module): # Check if 'weight' is pruned
                try:
                    prune.remove(module, 'weight')
                    pruned_count += 1
                except ValueError:
                    # This might happen if 'weight' exists but was not actually pruned
                    # or if there's some other issue with removing the hook.
                    self.logger.debug(f"Could not remove mask from {name}. It might not have been pruned or already removed.")
                except AttributeError: # If weight_mask or weight_orig is not found as expected by prune.remove
                     self.logger.debug(f"AttributeError while trying to remove mask from {name}. Parameter 'weight' might not be pruned.")

        self.logger.info(f"Removed pruning masks from {pruned_count} modules.")
    
    def fine_tune_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, sparsity: float):
        """Fine-tune a pruned model with checkpointing."""
        self.logger.info(f"Starting fine-tuning for {self.config.ft_epochs} epochs")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            # Filter out parameters that do not require gradients (e.g., frozen parts if any)
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.ft_learning_rate,
            momentum=self.config.ft_momentum,
            weight_decay=self.config.ft_weight_decay
        )
        
        ft_history = {
            'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'epoch_times': []
        }
        
        # Checkpoint paths
        ft_checkpoint_path = self.config.save_dir / f'ft_checkpoint_{sparsity*100:.0f}.pth'
        start_epoch = 0
        
        # Load fine-tuning checkpoint if exists
        if self.config.resume_pruning and ft_checkpoint_path.exists():
            try:
                checkpoint = torch.load(ft_checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                ft_history = checkpoint['history']
                self.logger.info(f"Resumed fine-tuning from epoch {start_epoch}")
            except Exception as e:
                self.logger.error(f"Error loading fine-tuning checkpoint: {e}. Starting fine-tuning from scratch.")
        
        for epoch in range(start_epoch, self.config.ft_epochs):
            epoch_start_time = time.time()
            model.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # Parameters with masks will have their gradients propagated to the original weights.
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                    self.logger.info(
                        f"FT Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] Batch Loss: {loss.item():.4f}"
                    )
            
            epoch_loss = running_loss / total if total > 0 else 0.0
            epoch_acc = correct / total if total > 0 else 0.0
            
            # Validation
            val_loss, val_acc = validate_model(model, val_loader, criterion, self.device)
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            ft_history['loss'].append(float(epoch_loss))
            ft_history['acc'].append(float(epoch_acc))
            ft_history['val_loss'].append(float(val_loss))
            ft_history['val_acc'].append(float(val_acc))
            ft_history['epoch_times'].append(float(epoch_time))
            
            self.logger.info(
                f"FT Epoch {epoch+1}/{self.config.ft_epochs} "
                f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save checkpoint every 5 epochs or at the end
            if (epoch + 1) % 5 == 0 or epoch == self.config.ft_epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': ft_history,
                    'sparsity': sparsity
                }
                torch.save(checkpoint, ft_checkpoint_path)
                self.logger.info(f"Fine-tuning checkpoint saved at epoch {epoch+1} to {ft_checkpoint_path}")
        
        # Clean up checkpoint file after successful completion of all epochs for this sparsity
        if ft_checkpoint_path.exists() and start_epoch < self.config.ft_epochs: # Only remove if FT completed
             if (self.config.ft_epochs - start_epoch) > 0: # Check if any epochs were run
                self.logger.info(f"Fine-tuning fully completed. Removing checkpoint: {ft_checkpoint_path}")
                ft_checkpoint_path.unlink()
        
        self.logger.info("Fine-tuning process finished.")
        return ft_history
    
    def evaluate_baseline(self, train_loader: DataLoader, val_loader: DataLoader, num_classes: int):
        """Evaluate baseline model for comparison."""
        self.logger.info("=== Evaluating Baseline Model ===")
        
        try:
            baseline_model = setup_model_structure(num_classes)
            baseline_state_dict = load_model_state_dict(Path(self.config.baseline_model_path), self.device)
            baseline_model.load_state_dict(baseline_state_dict)
            baseline_model = baseline_model.to(self.device)
            baseline_model.eval() # Set to eval mode
            
            criterion = nn.CrossEntropyLoss()
            base_val_loss, base_val_acc = validate_model(baseline_model, val_loader, criterion, self.device)
            base_size_mb = calculate_model_size(baseline_state_dict) # Use state_dict for more accurate size if masks not involved
            base_params = count_model_parameters(baseline_model)
            base_inference_metrics = measure_inference_speed(baseline_model, val_loader, self.device)
            
            self.logger.info(f"Baseline - Acc: {base_val_acc:.4f}, Val Loss: {base_val_loss:.4f}, Size: {base_size_mb:.2f} MB")
            self.logger.info(f"Baseline - Params (Total): {base_params['total_params']:,}, Speed: {base_inference_metrics.get('images_per_second', 0):.1f} img/s")
            
            self.baseline_results = {
                'run_type': 'baseline_re_eval',
                'model_path': str(self.config.baseline_model_path),
                'val_accuracy': float(base_val_acc),
                'val_loss': float(base_val_loss),
                'model_size_mb': float(base_size_mb),
                'parameter_counts': base_params,
                'inference_metrics': base_inference_metrics,
                'config': self.config.to_dict()
            }
            
            save_pruning_metrics(self.config.save_dir / 'baseline_re_eval_metrics.json', self.baseline_results)
            
        except Exception as e:
            self.logger.error(f"Error evaluating baseline: {e}")
            self.baseline_results = None # Ensure it's None if evaluation fails
    
    def run_pruning_experiment(self, sparsity: float, train_loader: DataLoader, val_loader: DataLoader, num_classes: int) -> Dict[str, Any]:
        """Run pruning experiment for a single sparsity level."""
        self.logger.info(f"\n=== Pruning Experiment: {sparsity*100:.1f}% Target Sparsity ===")
        
        try:
            # Load fresh baseline model for each sparsity experiment
            model = setup_model_structure(num_classes)
            state_dict = load_model_state_dict(Path(self.config.baseline_model_path), self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            
            params_before = count_model_parameters(model)
            self.logger.info(f"Parameters before pruning: Total={params_before['total_params']:,}, Non-Zero={params_before['non_zero_params']:,}")
            
            # Apply pruning
            self.apply_global_unstructured_pruning(model, sparsity)
            
            # Note: After `apply_global_unstructured_pruning`, the model has masks.
            # `count_model_parameters` should ideally reflect effective non-zero elements.
            params_after_masked = count_model_parameters(model)
            achieved_sparsity_masked = (params_before['total_params'] - params_after_masked['non_zero_params']) / params_before['total_params'] if params_before['total_params'] > 0 else 0
            self.logger.info(f"Achieved sparsity (with masks, counting zeros): {achieved_sparsity_masked*100:.2f}%")
            self.logger.info(f"Parameters after masking: Total={params_after_masked['total_params']:,}, Non-Zero (effective)={params_after_masked['non_zero_params']:,}")
            
            # Fine-tune
            ft_start_time = time.time()
            ft_history = self.fine_tune_model(model, train_loader, val_loader, sparsity)
            ft_total_time = time.time() - ft_start_time
            
            
            # Remove masks to make pruning permanent and potentially reduce model size
            self.remove_pruning_masks(model)
            if self.config.use_sparse_storage:
                model = convert_to_sparse(model)
                self.logger.info("Converted pruned weights to sparse tensors")
            else:
                self.logger.info("Sparse storage disabled in config")

            sparse_count = 0
            for name, tensor in model.state_dict().items():
                if tensor.is_sparse:
                    self.logger.info(f"Sparse tensor found: {name}")
                    sparse_count += 1
            self.logger.info(f"Total sparse tensors: {sparse_count}")
            
            # Final parameter count after masks are removed
            params_final = count_model_parameters(model) # Now counts actual non-zero weights
            achieved_sparsity_final = (params_before['total_params'] - params_final['non_zero_params']) / params_before['total_params'] if params_before['total_params'] > 0 else 0
            self.logger.info(f"Final physical sparsity (masks removed): {achieved_sparsity_final*100:.2f}%")
            self.logger.info(f"Parameters after mask removal: Total={params_final['total_params']:,}, Non-Zero={params_final['non_zero_params']:,}")
            
            # Save pruned model
            pruned_model_filename = f'resnet50_pruned_{sparsity*100:.0f}unstructured_ft.pth'
            pruned_model_path = self.config.save_dir / pruned_model_filename
            final_state_dict = model.state_dict() # Get state_dict after mask removal
            torch.save(final_state_dict, pruned_model_path)
            self.logger.info(f"Pruned model saved: {pruned_model_path}")
            
            # Final evaluation
            model.eval() # Set to eval mode
            criterion = nn.CrossEntropyLoss()
            final_val_loss, final_val_acc = validate_model(model, val_loader, criterion, self.device)
            # Model size calculation based on the saved state_dict (actual storage)
            
            final_size_dense = calculate_model_size(final_state_dict, count_sparse=False)
            final_size_sparse = calculate_model_size(final_state_dict, count_sparse=True)
            final_inference_metrics = measure_inference_speed(model, val_loader, self.device)
            
            self.logger.info(f"Final Evaluation - Acc: {final_val_acc:.4f}, Val Loss: {final_val_loss:.4f}")
            self.logger.info(f"Final Evaluation - Speed: {final_inference_metrics.get('images_per_second', 0):.1f} img/s")
            self.logger.info(f"Final Evaluation - Dense Size: {final_size_dense:.2f} MB, "
                 f"Sparse Size: {final_size_sparse:.2f} MB")
            
            # Compile results
            results = {
                'run_type': 'pruning_run',
                'pruning_method': self.config.pruning_method,
                'target_sparsity_rate': float(sparsity),
                'achieved_sparsity_masked_percent': float(achieved_sparsity_masked * 100),
                'achieved_sparsity_physical_percent': float(achieved_sparsity_final * 100),
                'pruned_model_saved_as': str(pruned_model_path),
                'fine_tuning_config': {
                    'epochs': self.config.ft_epochs,
                    'learning_rate': self.config.ft_learning_rate,
                    'momentum': self.config.ft_momentum,
                    'weight_decay': self.config.ft_weight_decay,
                    'total_time_seconds': float(ft_total_time),
                    'history': ft_history # Contains train/val acc/loss per epoch
                },
                'final_evaluation_metrics': {
                    'val_accuracy': float(final_val_acc),
                    'val_loss': float(final_val_loss),
                    'model_size_dense_mb': float(final_size_dense),
                    'model_size_sparse_mb': float(final_size_sparse),
                    'parameter_counts': params_final, # Parameters after mask removal
                    'inference_metrics': final_inference_metrics,
                },
                'pruning_config_snapshot': self.config.to_dict() # Save config for this run
            }
            
            results_file = self.config.save_dir / f'metrics_{sparsity*100:.0f}_unstructured.json'
            save_pruning_metrics(results_file, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pruning experiment for sparsity {sparsity*100:.1f}%: {e}", exc_info=True)
            return None

    def print_summary(self, all_results: List[Dict[str, Any]]):
        """Print comprehensive experiment summary."""
        self.logger.info("\n" + "="*100)
        self.logger.info("PRUNING EXPERIMENT SUMMARY")
        self.logger.info("="*100)

        baseline_speed = 1.0 

        if self.baseline_results:
            self.logger.info("\nBaseline Model:")
            b = self.baseline_results
            self.logger.info(
                f"  Accuracy: {b.get('val_accuracy', float('nan')):.4f} | "
                f"Loss: {b.get('val_loss', float('nan')):.4f} | "
                f"Size: {b.get('model_size_mb', float('nan')):.2f} MB"
            )
            param_counts_base = b.get('parameter_counts', {})
            inference_metrics_base = b.get('inference_metrics', {})
            self.logger.info(
                f"  Params (Total): {param_counts_base.get('total_params', 0):,} | "
                f"Params (Non-Zero): {param_counts_base.get('non_zero_params', 0):,} | "
                f"Speed: {inference_metrics_base.get('images_per_second', 0):.1f} img/s"
            )
            baseline_speed = inference_metrics_base.get('images_per_second', 1.0)
            if baseline_speed == 0: 
                baseline_speed = 1.0 # Avoid division by zero
        else:
            self.logger.info("\nBaseline model results not available.")
        
        if not all_results:
            self.logger.info("\nNo successful pruning runs completed or loaded to summarize.")
            self.logger.info("="*100)
            return
        
        all_results.sort(key=lambda x: x.get('target_sparsity_rate', 0))
        
        self.logger.info("\nPruning Results (Sorted by Target Sparsity):")
        self.logger.info("-" * 115) # Adjusted width for more columns
        self.logger.info(
                            f"{'Target SP%':<12} {'Achieved SP%':<14} {'Accuracy':<10} {'Loss':<8} "
                            f"{'Dense(MB)':<10} {'Sparse(MB)':<10} {'Params(NZ)':<12} {'Speed':<10}"
        )
        self.logger.info("-" * 115)

        
        for result in all_results:
            if 'final_evaluation_metrics' not in result or not isinstance(result['final_evaluation_metrics'], dict):
                self.logger.warning(
                    f"Skipping result for target sparsity {result.get('target_sparsity_rate', 'N/A')*100:.1f}% "
                    f"due to missing/malformed 'final_evaluation_metrics'."
                )
                continue

            metrics = result['final_evaluation_metrics']
            param_counts = metrics.get('parameter_counts', {})
            inference_metrics = metrics.get('inference_metrics', {})
            ft_config = result.get('fine_tuning_config', {})

            target_sparsity_pct = result.get('target_sparsity_rate', 0) * 100
            achieved_sparsity_pct = result.get('achieved_sparsity_physical_percent', float('nan'))

            val_accuracy = metrics.get('val_accuracy', float('nan'))
            val_loss = metrics.get('val_loss', float('nan'))
            model_size_mb = metrics.get('model_size_mb', float('nan'))
            non_zero_params = param_counts.get('non_zero_params', 0)
            
            speed = inference_metrics.get('images_per_second', 0)
            speedup = speed / baseline_speed if baseline_speed > 0 else 0.0
            ft_time_min = ft_config.get('total_time_seconds', 0) / 60.0
            
            dense_size = metrics.get('model_size_dense_mb', float('nan'))
            sparse_size = metrics.get('model_size_sparse_mb', float('nan'))

            self.logger.info(
                f"{target_sparsity_pct:>11.1f} "
                f"{achieved_sparsity_pct:>13.1f} "
                f"{val_accuracy:>9.4f} "
                f"{val_loss:>7.4f} "
                f"{dense_size:>9.2f} "      # Changed
                f"{sparse_size:>10.2f} "    # Changed
                f"{non_zero_params:>12,} "  # Changed
                f"{speed:>10.1f} "
                f"{speedup:>9.2f}x "
                f"{ft_time_min:>9.1f}"
            )
        self.logger.info("-" * 115)
        self.logger.info("="*100)

    def run(self):
        """Run the complete pruning experiment."""
        try:
            set_seed()
            self.logger.info("=== Starting Unstructured Pruning Experiment ===")
            self.logger_handler.log_config(self.config) # Log the PruningConfig
            
            train_loader, val_loader, num_classes = setup_data_loaders(self.config)
            self.logger_handler.log_dataset_info(
                len(train_loader.dataset), len(val_loader.dataset), num_classes
            )
            
            self.load_completed_experiments()
            
            self.evaluate_baseline(train_loader, val_loader, num_classes)
            
            all_results_for_summary = []
            
            # Add newly run experiments
            remaining_sparsities = [s for s in self.config.sparsity_rates if s not in self.completed_sparsities]
            if not remaining_sparsities and self.completed_sparsities:
                self.logger.info("All configured sparsity levels already have existing results. Skipping new runs.")
            elif not remaining_sparsities and not self.completed_sparsities:
                 self.logger.info("No sparsity levels to run and no completed experiments found.")
            else:
                self.logger.info(f"Running experiments for sparsities: {[f'{s*100:.1f}%' for s in remaining_sparsities]}")
                for sparsity in remaining_sparsities:
                    result = self.run_pruning_experiment(sparsity, train_loader, val_loader, num_classes)
                    if result:
                        all_results_for_summary.append(result)
            
            # Load existing results for the final summary
            if self.config.skip_completed: # Or always load for a full summary
                for sparsity_val in self.completed_sparsities:
                    # Avoid re-adding if it was just run (e.g., if skip_completed was False but resume was on)
                    # This logic assumes run_pruning_experiment handles saving its own results.
                    # And here we load any pre-existing ones not part of the current "run" session.
                    is_already_added = any(
                        r['target_sparsity_rate'] == sparsity_val for r in all_results_for_summary
                    )
                    if not is_already_added:
                        results_file = self.config.save_dir / f'metrics_{sparsity_val*100:.0f}_unstructured.json'
                        if results_file.exists():
                            try:
                                with open(results_file, 'r') as f:
                                    existing_result = json.load(f)
                                    all_results_for_summary.append(existing_result)
                            except Exception as e:
                                self.logger.warning(f"Could not load existing result file {results_file} for {sparsity_val*100:.1f}%: {e}")
                        else:
                             self.logger.warning(f"Expected results file {results_file} for completed sparsity {sparsity_val*100:.1f}% not found.")


            self.print_summary(all_results_for_summary)
            
            summary_data = {
                'experiment_type': 'unstructured_pruning',
                'baseline_results': self.baseline_results,
                'pruning_results_summary': all_results_for_summary, # List of dicts, one per sparsity
                'experiment_config': self.config.to_dict(),
                'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            summary_path = self.config.save_dir / 'complete_experiment_summary.json'
            save_pruning_metrics(summary_path, summary_data) # Assumes this util handles dicts and Path objects
            
            self.logger.info(f"\nExperiment completed! Full summary saved to {summary_path}")
            
        except KeyboardInterrupt:
            self.logger.info("\nExperiment interrupted by user.")
        except Exception as e:
            self.logger.error(f"Experiment failed critically: {e}", exc_info=True) # Log traceback
            # raise # Optionally re-raise if higher level handling is needed


def main():
    """Main entry point."""
    try:
        config = PruningConfig() # Can raise errors from __post_init__
        experiment = PruningExperiment(config)
        experiment.run()
    except FileNotFoundError as e:
        print(f"Configuration Error (FileNotFound): {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration Error (ValueError): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        # Potentially log to a fallback logger if TrainingLogger isn't initialized
        sys.exit(1)


if __name__ == '__main__':
    main()