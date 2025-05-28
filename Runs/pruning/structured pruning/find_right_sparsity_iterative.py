import torch
# import torch.nn as nn # No longer directly needed here if imported functions handle it
# from torchvision.models import resnet50 # Imported by prune.py
import os
import copy
import argparse
import json # For saving results

import torch_pruning as tp # Still needed if the imported fn doesn't re-export it

# --- Import functions from your main prune.py script ---
# Assuming prune.py is in the same directory
try:
    from prune import (
        get_device,
        count_parameters, # Assuming this is your version
        get_num_classes_from_data_dir, # Assuming this is your version
        apply_resnet50_structured_pruning_tp, # CRUCIAL: This is your working pruning function
        CONFIG as MAIN_SCRIPT_CONFIG # Optional: if you need any default values from it
    )
    print("Successfully imported functions from prune.py")
except ImportError as e:
    print(f"Error importing from prune.py: {e}")
    print("Please ensure prune.py is in the same directory and contains the required functions.")
    print("Required functions: get_device, count_parameters, get_num_classes_from_data_dir, apply_resnet50_structured_pruning_tp")
    exit()

# --- Configuration for this Iterative Search Script ---
ITERATIVE_SEARCH_CONFIG = {
    # Use data_dir and baseline_model_path from the main config if available, or override
    "data_dir": MAIN_SCRIPT_CONFIG.get("data_dir", "C:\\Uni\\deep_model_optimization\\imagenet-mini"),
    "baseline_model_path": MAIN_SCRIPT_CONFIG.get("baseline_model_path", "./best_model.pth"),
    "num_classes": MAIN_SCRIPT_CONFIG.get("num_classes", 1000), # Fallback, will try to auto-detect
    
    "iterative_overall_sparsity_targets": [0.50, 0.75, 0.90], 
    
    "per_step_test_rates": [
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75
    ],
    "output_rates_file": "iterative_structured_rates.json"
}

# --- Main Iterative Search Script Logic ---
def find_iterative_rates(config):
    device = get_device() # Imported
    print(f"Using device: {device}")

    num_classes_detected = get_num_classes_from_data_dir(config['data_dir']) # Imported
    if num_classes_detected is not None:
        config['num_classes'] = num_classes_detected
    print(f"Using num_classes: {config['num_classes']}")

    print(f"Loading baseline model from: {config['baseline_model_path']}")
    # Assuming resnet50 is imported via prune.py or available globally if prune.py imports it
    # If not, you might need: from torchvision.models import resnet50
    from torchvision.models import resnet50 # Explicitly import here for clarity if needed
    original_dense_model = resnet50(weights=None, num_classes=config['num_classes'])
    try:
        original_dense_model.load_state_dict(torch.load(config['baseline_model_path'], map_location=device))
    except Exception as e:
        print(f"Error loading baseline: {e}. Exiting.")
        return
    original_dense_model.to(device)
    original_dense_model.eval()

    original_dense_params = count_parameters(original_dense_model)['total_params'] # Imported
    print(f"Baseline model loaded. Original total parameters: {original_dense_params}")

    example_inputs = torch.randn(1, 3, 224, 224)

    chosen_rates_for_stages = []
    current_model_for_pruning = copy.deepcopy(original_dense_model)
    params_at_start_of_stage = original_dense_params

    for i, target_overall_sparsity in enumerate(config['iterative_overall_sparsity_targets']):
        stage_num = i + 1
        print(f"\n--- STAGE {stage_num}: Targeting ~{target_overall_sparsity*100:.1f}% Overall Sparsity (original: {original_dense_params}) ---")
        
        target_params_for_stage_end = original_dense_params * (1.0 - target_overall_sparsity)
        print(f"  Target parameters for end of Stage {stage_num}: ~{int(target_params_for_stage_end)}")
        print(f"  Parameters at start of Stage {stage_num}: {params_at_start_of_stage}")

        if params_at_start_of_stage <= target_params_for_stage_end + 1000: # Added a small tolerance
            print(f"  Model almost meets or exceeds target sparsity. Chosen per-step rate: 0.0 for Stage {stage_num}")
            chosen_rates_for_stages.append(0.0)
            continue

        print(f"  Searching for per-layer rate to apply to current model ({params_at_start_of_stage} params) to reach ~{int(target_params_for_stage_end)} params...")
        print("  | Test Per-Layer Rate for this step | Achieved Overall Sparsity (%) | Resulting Model Params |")

        stage_search_results = []
        best_rate_for_stage = 0.0 # Default to 0 if no pruning is better
        # Initialize min_param_diff with the difference if no pruning is done for this step
        min_param_diff = abs(params_at_start_of_stage - target_params_for_stage_end) 


        for per_step_rate in config['per_step_test_rates']:
            # If per_step_rate is 0, it means no pruning in this step.
            # This is handled by the best_rate_for_stage defaulting to 0 and min_param_diff initialization.
            if per_step_rate == 0 and params_at_start_of_stage > target_params_for_stage_end: # Only consider 0 if it's truly an option
                pass # Will be covered by initial min_param_diff if no other rate is better
            elif per_step_rate == 0 and params_at_start_of_stage <= target_params_for_stage_end:
                 continue # No need to test 0 if we already met target

            model_to_test_pruning_on = copy.deepcopy(current_model_for_pruning)
            
            # Use the imported pruning function
            pruned_test_model = apply_resnet50_structured_pruning_tp(
                model_to_test_pruning_on,
                example_inputs,
                target_pruning_rate_per_layer=per_step_rate,
                num_classes=config['num_classes'],
                device=device
            )

            resulting_model_params = count_parameters(pruned_test_model)['total_params']
            achieved_overall_sparsity_now = (original_dense_params - resulting_model_params) / original_dense_params if original_dense_params > 0 else 0
            
            print(f"  | {per_step_rate*100:<33.1f} | {achieved_overall_sparsity_now*100:<29.2f} | {resulting_model_params:<22} |")
            
            current_param_diff = abs(resulting_model_params - target_params_for_stage_end)
            if current_param_diff < min_param_diff:
                min_param_diff = current_param_diff
                best_rate_for_stage = per_step_rate
            # If we have the same diff, prefer smaller pruning rate (less aggressive)
            elif current_param_diff == min_param_diff and per_step_rate < best_rate_for_stage and best_rate_for_stage != 0 : # only if best_rate is not already 0
                 best_rate_for_stage = per_step_rate


        chosen_rates_for_stages.append(best_rate_for_stage)
        print(f"  For Stage {stage_num}, to reach overall target ~{target_overall_sparsity*100:.1f}%:")
        print(f"    Chosen per-layer rate for *this pruning step*: {best_rate_for_stage*100:.1f}%")

        if best_rate_for_stage > 0:
            print(f"    Applying chosen rate {best_rate_for_stage*100:.1f}% to model for next stage...")
            # Re-prune current_model_for_pruning with the best rate found for this stage
            current_model_for_pruning = apply_resnet50_structured_pruning_tp(
                current_model_for_pruning, # Prune the model that was at the start of this stage
                example_inputs,
                target_pruning_rate_per_layer=best_rate_for_stage,
                num_classes=config['num_classes'],
                device=device
            )
        
        params_at_start_of_stage = count_parameters(current_model_for_pruning)['total_params']
        final_overall_sparsity_this_stage = (original_dense_params - params_at_start_of_stage) / original_dense_params
        print(f"    Model after Stage {stage_num} pruning (using chosen step rate {best_rate_for_stage*100:.1f}%):")
        print(f"      Final Params: {params_at_start_of_stage}, Overall Sparsity from original: {final_overall_sparsity_this_stage*100:.2f}%")

    print("\n--- Iterative Pruning Rate Search Complete ---")
    print("Chosen per-layer rates to apply at each iterative step:")
    output_data = {"stages": []}
    for i, rate in enumerate(chosen_rates_for_stages):
        target_overall = config['iterative_overall_sparsity_targets'][i]
        stage_info = {
            "stage_number": i + 1,
            "target_overall_sparsity_approx": target_overall,
            "per_layer_rate_for_this_step": rate
        }
        print(f"  Stage {stage_info['stage_number']} (Target Overall ~{target_overall*100:.1f}%): Apply per-layer rate of {rate*100:.1f}% to model from prev. stage.")
        output_data["stages"].append(stage_info)
    
    with open(config['output_rates_file'], 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nChosen rates saved to {config['output_rates_file']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search script for iterative ResNet50 structured pruning rates.")
    # You can add CLI arguments to override ITERATIVE_SEARCH_CONFIG values if desired
    # e.g., parser.add_argument('--baseline_model_path', type=str)
    args = parser.parse_args()

    # Update config with CLI args if provided (example)
    # if args.baseline_model_path:
    #    ITERATIVE_SEARCH_CONFIG['baseline_model_path'] = args.baseline_model_path
    
    if not os.path.exists(ITERATIVE_SEARCH_CONFIG['data_dir']):
       print(f"Warning: data_dir '{ITERATIVE_SEARCH_CONFIG['data_dir']}' does not exist.")
    if not os.path.exists(ITERATIVE_SEARCH_CONFIG['baseline_model_path']):
       print(f"Error: baseline_model_path '{ITERATIVE_SEARCH_CONFIG['baseline_model_path']}' does not exist. Exiting.")
       exit()

    find_iterative_rates(ITERATIVE_SEARCH_CONFIG)