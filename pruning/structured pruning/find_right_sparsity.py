import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import os
import copy
import argparse

import torch_pruning as tp # Ensure torch-pruning is installed

# --- Configuration for this Search Script ---
# (You might need to adjust your main CONFIG paths if they are different)
SEARCH_CONFIG = {
    "data_dir": "C:\\Uni\\deep_model_optimization\\imagenet-mini", # Needed for num_classes, not for actual data loading here
    "baseline_model_path": "./best_model.pth", # Path to your trained baseline ResNet50
    "num_classes": 1000, # Default for ImageNet, adjust if your dataset/model is different
                         # This will be auto-detected if data_dir is set up correctly.
    # Define the per-layer rates you want to test to find overall sparsity
    "test_per_layer_rates": [
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85
    ],
    "desired_overall_sparsities_approx": [0.50, 0.75, 0.90] # For quick reference
}

# --- Helper Functions (Subset from your main script) ---

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return {"total_params": total_params, "non_zero_params": total_params} # For structurally pruned

def get_num_classes_from_data_dir(data_dir):
    try:
        train_path = os.path.join(data_dir, 'train')
        if os.path.exists(train_path) and os.path.isdir(train_path):
            num_classes = len(os.listdir(train_path))
            print(f"Auto-detected {num_classes} classes from {train_path}")
            return num_classes
        else:
            print(f"Warning: Train directory not found at {train_path}. Cannot auto-detect num_classes.")
            return None
    except Exception as e:
        print(f"Error auto-detecting num_classes: {e}")
        return None

# --- Your Adapted Pruning Function (apply_resnet50_structured_pruning_tp) ---
# Make sure this function is identical to the one in your main pruning script
# that uses torch-pruning, or copy it here. For brevity, I'm assuming it's defined
# elsewhere and can be imported or pasted. Here's a placeholder:

def apply_resnet50_structured_pruning_tp(model, example_inputs, target_pruning_rate_per_layer, num_classes, device):
    """
    Applies structured L1 filter pruning to ResNet50 Conv2D layers using torch-pruning.
    'target_pruning_rate_per_layer' is the fraction of filters to remove in each targeted Conv layer.
    This is a placeholder - ensure you use your actual working function.
    """
    print(f"\n--- Applying pruning with target_rate_per_layer: {target_pruning_rate_per_layer*100:.1f}% ---")
    # original_params_count = count_parameters(model)['total_params'] # Done outside in the loop
    model.to(device)
    
    ignored_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == num_classes:
            print(f"  Ignoring final classification layer: {name}")
            ignored_layers.append(m)

    importance = tp.importance.MagnitudeImportance(p=1, normalizer=None)
    
    pruner = tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=example_inputs.to(device),
        importance=importance,
        iterative_steps=1,
        pruning_ratio=target_pruning_rate_per_layer,
        global_pruning=False,
        ignored_layers=ignored_layers,
    )
    
    pruner.step() # Prune the model
    # Model is modified in-place
    return model


# --- Main Search Script Logic ---
def find_pruning_rates(config):
    device = get_device()
    print(f"Using device: {device}")

    # Determine num_classes
    num_classes = get_num_classes_from_data_dir(config['data_dir'])
    if num_classes is None:
        num_classes = config.get('num_classes', 1000) # Fallback to config or default
        print(f"Using num_classes: {num_classes} (fallback or from config)")
    else:
        config['num_classes'] = num_classes # Update config if auto-detected

    # Load baseline model
    print(f"Loading baseline model from: {config['baseline_model_path']}")
    baseline_model = resnet50(weights=None, num_classes=config['num_classes'])
    try:
        baseline_model.load_state_dict(torch.load(config['baseline_model_path'], map_location=device))
    except Exception as e:
        print(f"Error loading baseline model state_dict: {e}")
        print("Please ensure the baseline_model_path and num_classes are correct.")
        return
    baseline_model.to(device)
    baseline_model.eval() # Set to eval mode

    original_total_params = count_parameters(baseline_model)['total_params']
    print(f"Baseline model loaded. Original total parameters: {original_total_params}")

    # Example inputs for torch-pruning
    example_inputs = torch.randn(1, 3, 224, 224) # Will be moved to device in pruning_fn

    print("\nStarting search for per-layer rates to achieve desired overall sparsities...")
    print("Desired overall sparsities (approx): ", [f"{s*100:.1f}%" for s in config['desired_overall_sparsities_approx']])
    print("--------------------------------------------------------------------------")
    print("| Target Per-Layer Rate | Achieved Overall Parameter Sparsity (%) | Final Params |")
    print("|-----------------------|-----------------------------------------|--------------|")

    results = []

    for per_layer_rate in config['test_per_layer_rates']:
        # Create a fresh copy of the baseline model for each test
        current_model = copy.deepcopy(baseline_model)
        current_model.to(device) # Ensure it's on the device for pruning

        # Apply pruning
        pruned_model = apply_resnet50_structured_pruning_tp(
            current_model,
            example_inputs,
            target_pruning_rate_per_layer=per_layer_rate,
            num_classes=config['num_classes'],
            device=device
        )

        pruned_total_params = count_parameters(pruned_model)['total_params']
        achieved_overall_sparsity = (original_total_params - pruned_total_params) / original_total_params if original_total_params > 0 else 0
        
        print(f"| {per_layer_rate*100:<21.1f} | {achieved_overall_sparsity*100:<39.2f} | {pruned_total_params:<12} |")
        results.append({
            "per_layer_rate": per_layer_rate,
            "achieved_overall_sparsity": achieved_overall_sparsity,
            "final_params": pruned_total_params
        })

    print("--------------------------------------------------------------------------")
    print("\nSearch complete. Review the table above to select per-layer rates for your experiments.")
    print("For example, to get ~50% overall sparsity, look for a 'Target Per-Layer Rate' that results in 'Achieved Overall Sparsity' close to 50.00%.")
    
    # You can add logic here to suggest rates if you want, e.g., by finding the closest ones
    for target_overall_sparsity in config['desired_overall_sparsities_approx']:
        closest_result = min(results, key=lambda x: abs(x['achieved_overall_sparsity'] - target_overall_sparsity))
        print(f"  For target overall sparsity ~{target_overall_sparsity*100:.1f}%:")
        print(f"    Best found: Per-layer rate {closest_result['per_layer_rate']*100:.1f}% -> Achieved overall {closest_result['achieved_overall_sparsity']*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search script for ResNet50 structured pruning rates using torch-pruning.")
    parser.add_argument('--data_dir', type=str, default=SEARCH_CONFIG['data_dir'], help='Dataset directory (for num_classes detection)')
    parser.add_argument('--baseline_model_path', type=str, default=SEARCH_CONFIG['baseline_model_path'], help='Path to the baseline .pth model')
    # You can add an argument for test_per_layer_rates if you want to pass them via CLI as a comma-separated string

    args = parser.parse_args()
    
    SEARCH_CONFIG['data_dir'] = args.data_dir
    SEARCH_CONFIG['baseline_model_path'] = args.baseline_model_path
    
    # Basic check for data_dir existence
    if not os.path.exists(SEARCH_CONFIG['data_dir']):
       print(f"Warning: data_dir '{SEARCH_CONFIG['data_dir']}' does not exist. Num_classes detection might fail.")
    if not os.path.exists(SEARCH_CONFIG['baseline_model_path']):
       print(f"Error: baseline_model_path '{SEARCH_CONFIG['baseline_model_path']}' does not exist. Exiting.")
       exit()

    find_pruning_rates(SEARCH_CONFIG)