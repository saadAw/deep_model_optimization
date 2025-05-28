import torch
# import torch.nn as nn # Imported via prune.py
# from torchvision.models import resnet50 # Imported via prune.py
import os
import json
import time
import copy
import argparse
# import torch_pruning as tp # Imported via prune.py

# --- Import functions from your main prune.py script ---
try:
    from prune import (
        get_device,
        get_data_loaders,
        count_parameters,
        get_model_size_mb,
        evaluate_model,
        fine_tune_model, # Or use this if it handles the loop and optimizer creation
        apply_resnet50_structured_pruning_tp,
        calculate_achieved_parameter_sparsity,
        CONFIG as MAIN_SCRIPT_CONFIG, # Use for default paths, batch_size etc.
        resnet50 # Make sure resnet50 is importable or imported in prune.py
    )
except ImportError as e:
    print(f"Error importing from prune.py: {e}")
    print("Please ensure prune.py is in the same directory and contains the required functions/variables.")
    exit()

# --- Configuration for Iterative Pruning ---
ITERATIVE_CONFIG = {
    "data_dir": MAIN_SCRIPT_CONFIG.get("data_dir", "./imagenet-mini"),
    "save_dir": MAIN_SCRIPT_CONFIG.get("save_dir", "resnet50_pruning_iterative_structured_run"), # New save dir
    "baseline_model_path": MAIN_SCRIPT_CONFIG.get("baseline_model_path", "./best_model.pth"),
    "log_file_name": "iterative_structured_pruning_results.json",
    "batch_size": MAIN_SCRIPT_CONFIG.get("batch_size", 32),
    "num_workers": MAIN_SCRIPT_CONFIG.get("num_workers", 4),
    
    # Rates determined from your search_iterative_structured_rates.py script output
    # These are the per-layer rates to apply AT EACH STEP to the CURRENT model state
    "iterative_step_rates": [0.30, 0.30, 0.40], 
    # Corresponding fine-tuning epochs for each stage AFTER pruning
    "iterative_ft_epochs": [5, 10, 15], 
    # Target overall sparsities (approximate, for logging/reference) achieved after each stage
    "iterative_target_overall_sparsities_approx": [0.50, 0.75, 0.90],

    # Fine-tuning parameters (can also take from MAIN_SCRIPT_CONFIG if preferred)
    "ft_learning_rate": MAIN_SCRIPT_CONFIG.get("ft_learning_rate", 5e-5),
    "ft_momentum": MAIN_SCRIPT_CONFIG.get("ft_momentum", 0.9),
    "ft_weight_decay": MAIN_SCRIPT_CONFIG.get("ft_weight_decay", 1e-4),
}

# Helper to print sparsity information during iterative process
def print_iterative_sparsity_info(model, original_dense_params, stage_name, step_rate):
    current_params = count_parameters(model)['total_params']
    overall_sparsity = (original_dense_params - current_params) / original_dense_params if original_dense_params > 0 else 0
    print(f"\n{stage_name} (applied per-layer rate: {step_rate*100:.1f}% to previous model state):")
    print(f"  Current Total Params: {current_params}")
    print(f"  Overall Sparsity from Original Dense Model: {overall_sparsity*100:.2f}%")
    return current_params, overall_sparsity

# --- Main Iterative Pruning Script ---
def run_iterative_pruning(config):
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    device = get_device()
    print(f"Using device: {device}")

    # --- Data Loaders ---
    # num_classes will be determined here
    train_loader, val_loader, test_loader_inf, num_classes = get_data_loaders(
        config['data_dir'], config['batch_size'], config['num_workers']
    )
    print(f"Number of classes for the model: {num_classes}")

    # --- Criterion ---
    import torch.nn as nn # Explicit import if not exposed by prune.py
    criterion = nn.CrossEntropyLoss()

    # --- Load Baseline Model ---
    print(f"Loading baseline model from: {config['baseline_model_path']}")
    baseline_model_orig = resnet50(weights=None, num_classes=num_classes) # Ensure resnet50 is available
    try:
        baseline_model_orig.load_state_dict(torch.load(config['baseline_model_path'], map_location=device))
    except Exception as e:
        print(f"Error loading baseline model state_dict: {e}. Exiting.")
        return
    baseline_model_orig.to(device)
    
    # --- Evaluate Baseline (Optional but good for the log) ---
    print("\nEvaluating baseline model...")
    original_dense_params_count = count_parameters(baseline_model_orig)['total_params']
    baseline_eval_loss, baseline_eval_acc, baseline_inf_metrics = evaluate_model(
        baseline_model_orig, criterion, val_loader, device, measure_speed=True, test_loader_inf=test_loader_inf
    )
    baseline_size_mb = get_model_size_mb(baseline_model_orig)
    print(f"Baseline: Acc={baseline_eval_acc:.4f}, Loss={baseline_eval_loss:.4f}, Params={original_dense_params_count}, Size={baseline_size_mb:.2f}MB")

    # --- Results Logging Setup ---
    all_iterative_results = {
        "experiment_type": "iterative_structured_pruning_torch_pruning",
        "config": config,
        "baseline_metrics": {
            "accuracy": baseline_eval_acc,
            "loss": baseline_eval_loss,
            "params": original_dense_params_count,
            "size_mb": baseline_size_mb,
            "inference_metrics": baseline_inf_metrics
        },
        "iterative_stages": []
    }

    # --- Example Inputs for torch-pruning ---
    example_inputs = torch.randn(1, 3, 224, 224) # Device handled by pruning_fn

    # --- Iterative Pruning and Fine-tuning ---
    current_model_iter = copy.deepcopy(baseline_model_orig)
    
    if len(config['iterative_step_rates']) != len(config['iterative_ft_epochs']):
        print("Error: Mismatch between number of step rates and ft_epochs for iterative stages. Exiting.")
        return

    for i in range(len(config['iterative_step_rates'])):
        stage_num = i + 1
        step_rate = config['iterative_step_rates'][i]
        ft_epochs_for_stage = config['iterative_ft_epochs'][i]
        target_overall_sparsity_ref = config['iterative_target_overall_sparsities_approx'][i]

        print(f"\n===== ITERATIVE STAGE {stage_num} =====")
        print(f"Targeting overall sparsity ~{target_overall_sparsity_ref*100:.1f}% after this stage.")
        print(f"Applying per-layer rate of {step_rate*100:.1f}% to current model state.")

        # 1. Prune the current model
        current_model_iter = apply_resnet50_structured_pruning_tp(
            current_model_iter,
            example_inputs,
            target_pruning_rate_per_layer=step_rate,
            num_classes=num_classes,
            device=device
        )
        # Log sparsity after this pruning step
        params_after_prune_step, overall_sparsity_after_prune_step = print_iterative_sparsity_info(
            current_model_iter, original_dense_params_count, f"After Stage {stage_num} Pruning", step_rate
        )
        
        # Evaluate immediately after pruning (before FT for this stage)
        print(f"\nEvaluating Stage {stage_num} model immediately after pruning (before fine-tuning)...")
        eval_loss_before_ft, eval_acc_before_ft, _ = evaluate_model(
            current_model_iter, criterion, val_loader, device
        )
        size_mb_before_ft = get_model_size_mb(current_model_iter)
        print(f"  Acc (before FT): {eval_acc_before_ft:.4f}, Loss: {eval_loss_before_ft:.4f}, Size: {size_mb_before_ft:.2f}MB")

        # 2. Fine-tune the model for this stage
        # Create a temporary config for fine_tune_model if it expects a 'config' dict
        # with 'ft_epochs', 'ft_learning_rate', etc.
        temp_ft_config = {
            'ft_epochs': ft_epochs_for_stage,
            'ft_learning_rate': config['ft_learning_rate'],
            'ft_momentum': config['ft_momentum'],
            'ft_weight_decay': config['ft_weight_decay'],
            # Add any other keys your fine_tune_model expects from its 'config' arg
        }
        # The 'current_sparsity_rate' arg to fine_tune_model is a bit ambiguous here,
        # perhaps pass overall_sparsity_after_prune_step or stage_num
        print(f"\nFine-tuning Stage {stage_num} model ({ft_epochs_for_stage} epochs)...")
        ft_history, ft_time = fine_tune_model(
            current_model_iter, criterion, train_loader, val_loader, test_loader_inf, 
            device, temp_ft_config, overall_sparsity_after_prune_step # Pass overall sparsity for logging
        )

        # 3. Evaluate after fine-tuning for this stage
        print(f"\nEvaluating Stage {stage_num} model after fine-tuning...")
        final_eval_loss, final_eval_acc, final_inf_metrics = evaluate_model(
            current_model_iter, criterion, val_loader, device, measure_speed=True, test_loader_inf=test_loader_inf
        )
        final_params_stage = count_parameters(current_model_iter)['total_params']
        final_size_mb_stage = get_model_size_mb(current_model_iter)
        final_overall_sparsity_stage = calculate_achieved_parameter_sparsity(original_dense_params_count, final_params_stage)

        print(f"Stage {stage_num} Final: Acc={final_eval_acc:.4f}, Loss={final_eval_loss:.4f}, Params={final_params_stage}, Overall Sparsity={final_overall_sparsity_stage:.2f}%, Size={final_size_mb_stage:.2f}MB")
        if final_inf_metrics:
            print(f"  Inference: {final_inf_metrics.get('images_per_second',0):.2f} IPS, {final_inf_metrics.get('latency_ms_per_image',0):.2f} ms/img")

        # Save model state for this stage
        stage_model_filename = f"resnet50_iter_struct_stage{stage_num}_overall_sparsity_{int(final_overall_sparsity_stage)}_ft.pth"
        stage_model_save_path = os.path.join(config['save_dir'], stage_model_filename)
        torch.save(current_model_iter.state_dict(), stage_model_save_path)
        print(f"  Saved Stage {stage_num} model to {stage_model_save_path}")

        # Log results for this stage
        all_iterative_results["iterative_stages"].append({
            "stage_number": stage_num,
            "applied_step_rate": step_rate,
            "ft_epochs_this_stage": ft_epochs_for_stage,
            "target_overall_sparsity_approx": target_overall_sparsity_ref,
            "achieved_overall_sparsity_percent": final_overall_sparsity_stage,
            "params_after_pruning_this_step": params_after_prune_step,
            "metrics_before_ft_this_stage": {
                "accuracy": eval_acc_before_ft,
                "loss": eval_loss_before_ft,
                "size_mb": size_mb_before_ft,
            },
            "fine_tuning_details": {
                "total_time_seconds": ft_time,
                "history": ft_history # Ensure ft_history is serializable (list of floats)
            },
            "final_metrics_this_stage": {
                "accuracy": final_eval_acc,
                "loss": final_eval_loss,
                "params": final_params_stage,
                "size_mb": final_size_mb_stage,
                "inference_metrics": final_inf_metrics
            },
            "model_saved_path": stage_model_save_path
        })
        
        # Save results incrementally
        with open(os.path.join(config['save_dir'], config['log_file_name']), 'w') as f:
            json.dump(all_iterative_results, f, indent=2)
        print(f"  Incrementally saved results to {config['log_file_name']}")

    print("\n===== ITERATIVE STRUCTURED PRUNING COMPLETE =====")
    print(f"All results saved to {os.path.join(config['save_dir'], config['log_file_name'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Iterative Structured Pruning for ResNet50 with torch-pruning")
    # Add CLI arguments if you want to override ITERATIVE_CONFIG values
    # e.g., parser.add_argument('--save_dir', type=str)
    args = parser.parse_args() # Basic, no CLI args defined here

    # You can update ITERATIVE_CONFIG with args here if needed
    # if args.save_dir: ITERATIVE_CONFIG['save_dir'] = args.save_dir
    
    # Basic path checks
    if not os.path.exists(ITERATIVE_CONFIG['data_dir']):
       print(f"Warning: data_dir '{ITERATIVE_CONFIG['data_dir']}' does not exist.")
    if not os.path.exists(ITERATIVE_CONFIG['baseline_model_path']):
       print(f"Error: baseline_model_path '{ITERATIVE_CONFIG['baseline_model_path']}' does not exist. Exiting.")
       exit()

    run_iterative_pruning(ITERATIVE_CONFIG)