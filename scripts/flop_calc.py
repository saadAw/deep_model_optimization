import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile # For FLOPs calculation
from pathlib import Path
import glob
import gc
import re # For parsing stage numbers

# Ensure torch-pruning is installed: pip install torch-pruning
try:
    import torch_pruning as tp
except ImportError:
    print("ERROR: torch-pruning library not found. Please install it: pip install torch-pruning")
    exit()

# --- Configuration ---
MODELS_ROOT_DIR = "saved_models_and_logs"  # Adjust to your main experiments folder
ONE_SHOT_STRUCTURED_MODEL_DIRS = [
    os.path.join(MODELS_ROOT_DIR, "pruning_structured_oneshot"),
]
ITERATIVE_STRUCTURED_MODEL_DIRS = [
    os.path.join(MODELS_ROOT_DIR, "pruning_structured_iterative"),
]

OUTPUT_CSV = "model_flops_params_summary_fixed_classes.csv"
FIXED_NUM_CLASSES = 1000 # All experiments use 1000 classes
INPUT_TENSOR_CPU = torch.randn(1, 3, 224, 224) # For thop
DEVICE = torch.device("cpu") # FLOPs calculation is done on CPU

print(f"--- FLOPs/Params Calculation Script ---")
print(f"MODELS_ROOT_DIR: {Path(MODELS_ROOT_DIR).resolve()}")
print(f"Output CSV: {Path(OUTPUT_CSV).resolve()}")
print(f"FIXED_NUM_CLASSES: {FIXED_NUM_CLASSES}")

# --- Model Definition and Pruning Application Helpers ---
def get_base_model(arch_name="resnet50"):
    """Instantiates a base model (ResNet18 or ResNet50) with FIXED_NUM_CLASSES."""
    arch_name_low = arch_name.lower()
    if arch_name_low == "resnet50":
        model = models.resnet50(weights=None, num_classes=FIXED_NUM_CLASSES)
    elif arch_name_low == "resnet18":
        model = models.resnet18(weights=None, num_classes=FIXED_NUM_CLASSES)
    else:
        print(f"Warning: Base architecture '{arch_name}' not recognized. Defaulting to ResNet50.")
        model = models.resnet50(weights=None, num_classes=FIXED_NUM_CLASSES)
    return model

def apply_structured_pruning_for_reconstruction(
    model_to_prune, example_inputs, target_pruning_rate_per_layer, device
):
    """Applies structured pruning. Assumes FIXED_NUM_CLASSES for FC layer."""
    model_to_prune.to(device)
    example_inputs = example_inputs.to(device)
    ignored_layers = []
    for name, m in model_to_prune.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == FIXED_NUM_CLASSES:
            ignored_layers.append(m)
            # print(f"    Ignoring FC layer for pruning: {name} with out_features={m.out_features}")

    importance = tp.importance.MagnitudeImportance(p=1)
    
    # Fixed API call - removed deprecated parameters
    pruner = tp.pruner.MagnitudePruner(
        model=model_to_prune, 
        example_inputs=example_inputs, 
        importance=importance,
        iterative_steps=1, 
        pruning_ratio=target_pruning_rate_per_layer,
        global_pruning=False, 
        ignored_layers=ignored_layers
        # Removed: customized_pruning_ops=None, unwrapped_parameters=None
    )
    pruner.step() # Modifies model in-place
    return model_to_prune

# --- Log Parsing Helper (Only for pruning config) ---
def get_pruning_config_from_log(log_file_path):
    if not Path(log_file_path).exists():
        return None
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
        
        config_details = log_data.get('config_details', {})
        if 'target_filter_pruning_rate_per_layer' in config_details:
            rate = config_details['target_filter_pruning_rate_per_layer']
            return {'type': 'one-shot', 'rate': float(rate)}
        
        if 'applied_step_rate_for_this_stage' in config_details:
            rate = config_details['applied_step_rate_for_this_stage']
            return {'type': 'iterative_step', 'rate': float(rate)}
    except Exception:
        pass
    return None

# --- FLOPs Calculation Functions ---
def reconstruct_and_calculate_flops_structured(
    model_path_str, pruning_config_for_reconstruction, base_arch="resnet50"
):
    model_path = Path(model_path_str)
    results = {'flops_gmacs': "N/A", 'params_millions': "N/A", 'load_status': "Not Attempted (Structured)"}
    
    reconstructed_model = get_base_model(base_arch) # Uses FIXED_NUM_CLASSES
    reconstructed_model.to(DEVICE)
    example_inputs_for_pruning = torch.randn(1, 3, 224, 224).to(DEVICE)

    try:
        if pruning_config_for_reconstruction['type'] == 'one-shot':
            rate = pruning_config_for_reconstruction['rate']
            reconstructed_model = apply_structured_pruning_for_reconstruction(
                reconstructed_model, example_inputs_for_pruning, rate, DEVICE)
        elif pruning_config_for_reconstruction['type'] == 'iterative':
            step_rates = pruning_config_for_reconstruction['step_rates']
            current_arch_model = reconstructed_model
            for _, step_rate in enumerate(step_rates):
                current_arch_model = apply_structured_pruning_for_reconstruction(
                    current_arch_model, example_inputs_for_pruning, step_rate, DEVICE)
            reconstructed_model = current_arch_model
        else:
            results['load_status'] = f"Failed (Unknown pruning_config type: {pruning_config_for_reconstruction['type']})"
            return results

        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        reconstructed_model.load_state_dict(state_dict, strict=True)
        results['load_status'] = "Success (Structured Reconstructed)"
        reconstructed_model.eval()

        model_for_flops = reconstructed_model.to(torch.device('cpu'))
        dummy_input_thop = INPUT_TENSOR_CPU.to(torch.device('cpu'))
        macs, params = profile(model_for_flops, inputs=(dummy_input_thop,), verbose=False)
        results['flops_gmacs'] = macs / 1e9
        results['params_millions'] = params / 1e6
        del model_for_flops
    except Exception as e:
        results['load_status'] = f"Failed (Structured Reconstruction Error: {str(e).splitlines()[0]})"
        # import traceback; traceback.print_exc() # Uncomment for detailed errors
    
    if 'reconstructed_model' in locals(): del reconstructed_model
    gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache() # Though DEVICE is CPU, good practice
    return results

def calculate_flops_for_general_model(
    model_path_str, base_arch="resnet50"
):
    model_path = Path(model_path_str)
    results = {'flops_gmacs': "N/A", 'params_millions': "N/A", 'load_status': "Not Attempted (General)"}
    
    loaded_model = None
    is_jit_model = False

    try:
        try:
            loaded_model = torch.jit.load(model_path, map_location=DEVICE)
            is_jit_model = True
            results['load_status'] = "Success (JIT)"
        except Exception:
            loaded_model = None

        if loaded_model is None:
            _loaded_content = torch.load(model_path, map_location=DEVICE, weights_only=False)
            if isinstance(_loaded_content, nn.Module):
                loaded_model = _loaded_content
                results['load_status'] = "Success (Full Model)"
            elif isinstance(_loaded_content, dict):
                model_instance = get_base_model(base_arch) # Uses FIXED_NUM_CLASSES
                state_dict = _loaded_content
                if 'model' in state_dict and isinstance(state_dict['model'], dict): state_dict = state_dict['model']
                elif 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict): state_dict = state_dict['state_dict']
                if all(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Ensure the loaded state_dict is compatible with a FIXED_NUM_CLASSES model
                # This might involve checking/adjusting the FC layer keys if the saved model had different classes.
                # However, since you confirmed all are 1000 classes, this should be less of an issue.
                # If size mismatch occurs, it will be caught by load_state_dict.
                model_instance.load_state_dict(state_dict, strict=True)
                loaded_model = model_instance
                results['load_status'] = "Success (StateDict)"
            else:
                raise RuntimeError(f"Loaded content is not nn.Module or dict: {type(_loaded_content)}")
        
        if loaded_model is None:
            results['load_status'] = "Failed (Unknown Load Type)"
            return results

        loaded_model.to(DEVICE).eval()

        if is_jit_model:
            results['flops_gmacs'] = "N/A (JIT Model)"
            results['params_millions'] = "N/A (JIT Model)"
            # Note: For JIT, if the original nn.Module definition is available,
            # you could instantiate it, load weights (if possible to extract), and then profile.
            # But given just a .pth, this is the common limitation.
        else:
            model_for_flops = loaded_model.to(torch.device('cpu'))
            dummy_input_thop = INPUT_TENSOR_CPU.to(torch.device('cpu'))
            macs, params = profile(model_for_flops, inputs=(dummy_input_thop,), verbose=False)
            results['flops_gmacs'] = macs / 1e9
            results['params_millions'] = params / 1e6
            del model_for_flops
    except Exception as e:
        results['load_status'] = f"Failed (General Load/Profile Error: {str(e).splitlines()[0]})"
        # import traceback; traceback.print_exc() # Uncomment for detailed errors

    if loaded_model: del loaded_model
    gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return results

# --- Main Script ---
if __name__ == "__main__":
    all_model_metrics = []
    processed_model_files = set() 

    all_potential_exp_dirs = []
    for root, _, files in os.walk(MODELS_ROOT_DIR):
        if any(f.endswith(".pth") for f in files):
            all_potential_exp_dirs.append(Path(root))
    all_potential_exp_dirs = sorted(list(set(all_potential_exp_dirs)))
    print(f"Found {len(all_potential_exp_dirs)} potential experiment directories with .pth files.")

    def get_experiment_base_arch(exp_path_obj):
        return "resnet18" if "resnet18" in exp_path_obj.name.lower() else "resnet50"

    def process_model_file(model_file_path_obj, exp_path_obj, handler_type, pruning_rec_config_override=None):
        if str(model_file_path_obj.resolve()) in processed_model_files:
            return None 
        
        print(f"\n({handler_type}) Processing: {exp_path_obj}")
        print(f"  Model file: {model_file_path_obj.name}")
        
        log_file_path = exp_path_obj / "log.json"
        experiment_id = f"{exp_path_obj.parent.name}/{exp_path_obj.name}"
        base_arch = get_experiment_base_arch(exp_path_obj)
        metrics = {}

        if handler_type == "Structured": # Handles both one-shot and iterative if config is provided
            pruning_config_to_use = pruning_rec_config_override or get_pruning_config_from_log(log_file_path)
            if pruning_config_to_use:
                print(f"    Reconstruction config (Structured): {pruning_config_to_use}")
                metrics = reconstruct_and_calculate_flops_structured(
                    str(model_file_path_obj), pruning_config_to_use, base_arch
                )
            else:
                print(f"    Structured config issue. Falling back to general load for {model_file_path_obj.name}.")
                metrics = calculate_flops_for_general_model(str(model_file_path_obj), base_arch)
        else: # General handler
            metrics = calculate_flops_for_general_model(str(model_file_path_obj), base_arch)
        
        processed_model_files.add(str(model_file_path_obj.resolve()))
        result_dict = {
            'Experiment_ID': experiment_id, 'Model_Path': str(model_file_path_obj), **metrics
        }
        print(f"    Finished: {metrics.get('load_status', 'N/A')}. FLOPs: {metrics.get('flops_gmacs', 'N/A')}, Params: {metrics.get('params_millions', 'N/A')}")
        return result_dict

    # --- Process One-Shot Structured Pruning Experiments ---
    print("\n--- Processing One-Shot Structured Pruning Experiments ---")
    for dir_pattern_list in ONE_SHOT_STRUCTURED_MODEL_DIRS:
        for exp_dir_globbed in glob.glob(os.path.join(dir_pattern_list, "*")):
            exp_path = Path(exp_dir_globbed)
            if exp_path.is_dir():
                model_file_pth = next(exp_path.glob("model_final.pth"), next(exp_path.glob("*.pth"), None))
                if model_file_pth:
                    result = process_model_file(model_file_pth, exp_path, "Structured")
                    if result: all_model_metrics.append(result)
    
    # --- Process Iterative Structured Pruning Experiments ---
    print("\n--- Processing Iterative Structured Pruning Experiments ---")
    iterative_exp_groups = {}
    for dir_pattern_list in ITERATIVE_STRUCTURED_MODEL_DIRS:
        for exp_dir_globbed in glob.glob(os.path.join(dir_pattern_list, "*")):
            exp_path = Path(exp_dir_globbed)
            if exp_path.is_dir():
                match = re.search(r"(stage(\d+))", exp_path.name.lower())
                if match:
                    stage_num = int(match.group(2))
                    base_name_parts = exp_path.name.lower().split(match.group(1))
                    base_name_key = f"{exp_path.parent.name}_{base_name_parts[0].rstrip('_') if base_name_parts[0] else 'iter_exp'}"
                    if base_name_key not in iterative_exp_groups: iterative_exp_groups[base_name_key] = []
                    iterative_exp_groups[base_name_key].append({'path': exp_path, 'stage': stage_num})

    for base_name_key, stages_info in iterative_exp_groups.items():
        print(f"\nProcessing Iterative Group: {base_name_key}")
        sorted_stages = sorted(stages_info, key=lambda x: x['stage'])
        cumulative_step_rates = []
        for stage_info in sorted_stages:
            exp_path = stage_info['path']
            current_stage_num = stage_info['stage']
            
            model_file_pth = next(exp_path.glob("model_final.pth"), next(exp_path.glob("*.pth"), None))
            if not model_file_pth:
                print(f"  Model file not found in {exp_path} (Stage {current_stage_num}). Skipping.")
                all_model_metrics.append({
                    'Experiment_ID': f"{exp_path.parent.name}/{exp_path.name}", 'Model_Path': "N/A (Iterative Stage)",
                    'flops_gmacs': "N/A", 'params_millions': "N/A", 'load_status': "No Model File"
                }); continue

            log_file_path = exp_path / "log.json"
            current_stage_pruning_info = get_pruning_config_from_log(log_file_path)
            
            if current_stage_pruning_info and current_stage_pruning_info.get('type') == 'iterative_step':
                cumulative_step_rates.append(current_stage_pruning_info['rate'])
                pruning_rec_config = {'type': 'iterative', 'step_rates': list(cumulative_step_rates)}
                result = process_model_file(model_file_pth, exp_path, "Structured", pruning_rec_config_override=pruning_rec_config)
                if result: all_model_metrics.append(result)
            else:
                print(f"      Iterative step rate not found for Stage {current_stage_num}. Fallback to general.")
                result = process_model_file(model_file_pth, exp_path, "General (Fallback)")
                if result: all_model_metrics.append(result)

    # --- Process all other experiments (general fallback) ---
    print("\n--- Processing Remaining Experiments (General Fallback) ---")
    for exp_path in all_potential_exp_dirs:
        # This loop will try to process any .pth file in any directory found by os.walk
        # if it hasn't been caught by the specific handlers above.
        model_file_pth = next(exp_path.glob("model_final.pth"), next(exp_path.glob("*.pth"), None))
        if model_file_pth:
            result = process_model_file(model_file_pth, exp_path, "General (Fallback)")
            if result: all_model_metrics.append(result)

    # --- Create DataFrame and Save ---
    if not all_model_metrics:
        print("\nNo models were processed. Output CSV will be empty or not created.")
    else:
        df_temp = pd.DataFrame(all_model_metrics)
        df_results = df_temp.sort_values(by=['Model_Path', 'load_status'], ascending=[True, True])
        df_results.drop_duplicates(subset=['Model_Path'], keep='first', inplace=True)
        df_results = df_results.sort_values(by=['Experiment_ID'])

        cols = ['Experiment_ID', 'Model_Path', 'flops_gmacs', 'params_millions', 'load_status']
        df_results = df_results.reindex(columns=cols) 
        df_results.to_csv(OUTPUT_CSV, index=False, lineterminator='\n', float_format='%.5f')
        print(f"\n--- FLOPs and Params summary saved to {OUTPUT_CSV} ---")
        if not df_results.empty:
            print("\nSample of the summary (first 5 rows):")
            print(df_results.head().to_string())
        else:
            print("DataFrame is empty.")

    if all_model_metrics and 'df_results' in locals() and not df_results.empty:
        print(f"\nTotal unique model files processed: {len(df_results)}")
        successful_loads = sum(1 for r_status in df_results['load_status'] if "Success" in str(r_status))
        print(f"Models successfully loaded for FLOPs/Params calculation: {successful_loads}")
    elif all_model_metrics:
        print("\nSome models processed but DataFrame construction failed or was empty.")
    else:
        print("\nNo models processed.")
    print("--- FLOPs Calculation Script Finished ---")