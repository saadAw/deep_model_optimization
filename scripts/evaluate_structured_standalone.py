import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
import glob
import gc
from thop import profile
from pathlib import Path
import torch_pruning as tp
import re # For parsing stage numbers

# --- Configuration (Identical to previous) ---
MODELS_ROOT_DIR = "saved_models_and_logs"
ONE_SHOT_STRUCTURED_MODEL_DIRS = [
    "saved_models_and_logs/pruning_structured_oneshot",
]
ITERATIVE_STRUCTURED_MODEL_DIRS = [
    "saved_models_and_logs/pruning_structured_iterative",
]
OUTPUT_CSV = "structured_pruning_evaluation_summary_standalone.csv"
FIXED_NUM_CLASSES = 1000
VALIDATION_DATA_PATH = "imagenet-mini/val"
BATCH_SIZE_EVAL = 32
NUM_WORKERS_EVAL = 0
MAX_EVAL_BATCHES = 125 # Set high for robust accuracy, low for testing
INPUT_TENSOR_CPU = torch.randn(1, 3, 224, 224)
INPUT_TENSOR_GPU = INPUT_TENSOR_CPU.cuda() if torch.cuda.is_available() else None
WARMUP_INFERENCES = 5
TIMED_INFERENCES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Standalone Evaluation Script Starting on Device: {DEVICE} ---")
print(f"--- Using FIXED_NUM_CLASSES = {FIXED_NUM_CLASSES} ---")
print(f"--- Ensure 'torch-pruning' library is installed! ---")

# --- Model Definition and Pruning Application (Identical) ---
def get_base_resnet50_model():
    model = models.resnet50(weights=None, num_classes=FIXED_NUM_CLASSES)
    return model

def apply_structured_pruning_to_model_for_reconstruction(
    model_to_prune, example_inputs, target_pruning_rate_per_layer, device
):
    # print(f"    Applying torch-pruning: Target rate {target_pruning_rate_per_layer*100:.1f}% for arch reconstruction.") # Verbose
    model_to_prune.to(device)
    example_inputs = example_inputs.to(device)
    ignored_layers = []
    for name, m in model_to_prune.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == FIXED_NUM_CLASSES:
            ignored_layers.append(m)
    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model=model_to_prune, example_inputs=example_inputs, importance=importance,
        iterative_steps=1, pruning_ratio=target_pruning_rate_per_layer,
        global_pruning=False, ignored_layers=ignored_layers,
    )
    pruner.step()
    return model_to_prune

# --- Evaluation Helpers (Identical) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
eval_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,
])

@torch.no_grad()
def evaluate_model_accuracy(model, device_str, max_batches_to_eval, experiment_id_for_log=""):
    if not os.path.exists(VALIDATION_DATA_PATH):
        print(f"ERROR ({experiment_id_for_log}): Val data path not found: {VALIDATION_DATA_PATH}")
        return "N/A (Val Data Missing)"
    try:
        val_dataset = ImageFolder(VALIDATION_DATA_PATH, eval_transforms)
        if len(val_dataset.classes) != FIXED_NUM_CLASSES:
             print(f"WARNING ({experiment_id_for_log}): Val Dataset classes ({len(val_dataset.classes)}) != FIXED_NUM_CLASSES ({FIXED_NUM_CLASSES}).")
        if len(val_dataset) == 0: return 0.0
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False,
                                num_workers=NUM_WORKERS_EVAL, pin_memory=(device_str=='cuda'))
    except Exception as e:
        print(f"ERROR ({experiment_id_for_log}): Could not load val data: {e}")
        return f"N/A (Val Data Load Error: {str(e).splitlines()[0]})"
    device_obj = torch.device(device_str)
    model.to(device_obj); model.eval()
    correct, total, batches_processed = 0,0,0
    for images, labels in val_loader:
        images, labels = images.to(device_obj), labels.to(device_obj)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batches_processed += 1
        if batches_processed >= max_batches_to_eval: break
    accuracy = (correct / total) if total > 0 else 0.0
    print(f"      INFO ({experiment_id_for_log}): Accuracy = {accuracy:.4f} ({correct}/{total}) on {batches_processed} batches.")
    return accuracy

def load_reconstructed_pruned_model(model_path, pruning_config, device_obj):
    # print(f"    Attempting to load pruned model: {model_path}") # Verbose
    # print(f"    Using pruning config for reconstruction: {pruning_config}") # Verbose
    reconstructed_model = get_base_resnet50_model()
    reconstructed_model.to(device_obj)
    example_inputs = torch.randn(1, 3, 224, 224).to(device_obj)
    try:
        if pruning_config['type'] == 'one-shot':
            rate = pruning_config['rate']
            reconstructed_model = apply_structured_pruning_to_model_for_reconstruction(
                reconstructed_model, example_inputs, rate, device_obj)
        elif pruning_config['type'] == 'iterative':
            step_rates = pruning_config['step_rates']
            current_arch_model = reconstructed_model
            for i, step_rate in enumerate(step_rates):
                current_arch_model = apply_structured_pruning_to_model_for_reconstruction(
                    current_arch_model, example_inputs, step_rate, device_obj)
            reconstructed_model = current_arch_model
        else:
            print(f"    ERROR: Unknown pruning_config type: {pruning_config['type']}")
            return None
        state_dict = torch.load(model_path, map_location=device_obj)
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        reconstructed_model.load_state_dict(state_dict)
        print(f"    State_dict loaded successfully into reconstructed model: {model_path}")
        reconstructed_model.eval()
        return reconstructed_model
    except Exception as e:
        print(f"    ERROR loading/reconstructing pruned model {model_path}: {e}")
        import traceback; traceback.print_exc()
        return None

def process_single_structured_pruned_model(model_path, experiment_id, pruning_config_for_reconstruction):
    # (Mostly Identical to previous, only calls to helpers changed slightly if needed)
    results = {
        'Experiment_ID': experiment_id, 'Model_Path': model_path, 'Num_Classes': FIXED_NUM_CLASSES,
        'Pruning_Type': pruning_config_for_reconstruction.get('type', 'N/A'),
        'Pruning_Params_Detail': str(pruning_config_for_reconstruction), 'Model_Size_MB_Disk': "N/A",
        'Final_Val_Accuracy': "N/A (Processing Error)", 'Inference_Time_ms_CPU (Batch 1)': "N/A",
        'Inference_Time_ms_GPU (Batch 1)': "N/A (CUDA unavailable or error)",
        'FLOPs_GMACs': "N/A", 'Params_Millions': "N/A", 'Load_Status': "Failed"
    }
    if not model_path or not os.path.exists(model_path):
        results['Final_Val_Accuracy'] = "N/A (No model file)"; return results
    results['Model_Size_MB_Disk'] = os.path.getsize(model_path) / (1024 * 1024)
    loaded_model = load_reconstructed_pruned_model(model_path, pruning_config_for_reconstruction, DEVICE)
    if loaded_model is None:
        results['Final_Val_Accuracy'] = "N/A (Model Load/Reconstruction Error)"; return results
    results['Load_Status'] = "Success"
    try:
        model_for_flops_params = loaded_model.to(torch.device('cpu'))
        dummy_input_flops = INPUT_TENSOR_CPU.to(torch.device('cpu'))
        macs, params = profile(model_for_flops_params, inputs=(dummy_input_flops,), verbose=False)
        results['FLOPs_GMACs'] = macs / 1e9; results['Params_Millions'] = params / 1e6
        del model_for_flops_params
    except Exception as e_flops:
        print(f"      WARNING ({experiment_id}): FLOPs/Params error: {e_flops}")
        results['FLOPs_GMACs'] = "N/A (Thop Error)"; results['Params_Millions'] = "N/A (Thop Error)"
    accuracy_val = evaluate_model_accuracy(loaded_model, DEVICE.type, MAX_EVAL_BATCHES, experiment_id)
    results['Final_Val_Accuracy'] = accuracy_val
    cpu_model_for_timing = loaded_model.to(torch.device('cpu'))
    try:
        with torch.no_grad():
            for _ in range(WARMUP_INFERENCES): _ = cpu_model_for_timing(INPUT_TENSOR_CPU)
            timings_cpu = []
            for _ in range(TIMED_INFERENCES):
                start_time = time.perf_counter()
                _ = cpu_model_for_timing(INPUT_TENSOR_CPU)
                end_time = time.perf_counter()
                timings_cpu.append((end_time - start_time) * 1000) # Multiply by 1000 here for ms
        results['Inference_Time_ms_CPU (Batch 1)'] = sum(timings_cpu) / len(timings_cpu) if timings_cpu else "N/A"
    except Exception as e_cpu_time:
        results['Inference_Time_ms_CPU (Batch 1)'] = f"N/A (CPU Time Error: {str(e_cpu_time).splitlines()[0]})"
    if 'cpu_model_for_timing' in locals(): del cpu_model_for_timing
    if DEVICE.type == 'cuda' and INPUT_TENSOR_GPU is not None:
        gpu_model_for_timing = loaded_model.to(torch.device('cuda'))
        try:
            with torch.no_grad():
                for _ in range(WARMUP_INFERENCES): _ = gpu_model_for_timing(INPUT_TENSOR_GPU); torch.cuda.synchronize()
                timings_gpu = []
                for _ in range(TIMED_INFERENCES):
                    torch.cuda.synchronize(); t0 = time.perf_counter(); _ = gpu_model_for_timing(INPUT_TENSOR_GPU); torch.cuda.synchronize(); t1 = time.perf_counter()
                    timings_gpu.append((t1 - t0) * 1000)
            results['Inference_Time_ms_GPU (Batch 1)'] = sum(timings_gpu) / len(timings_gpu) if timings_gpu else "N/A"
        except Exception as e_gpu_time: results['Inference_Time_ms_GPU (Batch 1)'] = f"N/A (GPU Time Error: {str(e_gpu_time).splitlines()[0]})"
        if 'gpu_model_for_timing' in locals(): del gpu_model_for_timing
    elif DEVICE.type != 'cuda': results['Inference_Time_ms_GPU (Batch 1)'] = "N/A (CUDA unavailable)"
    del loaded_model
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    return results

# --- REVISED Main Logic to Find Models and Get Pruning Configs ---
def get_pruning_config_from_log(log_file_path):
    """Helper to load log and extract key pruning param for a single stage/one-shot."""
    if not log_file_path.exists():
        # print(f"    Log file not found: {log_file_path}") # Verbose
        return None
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
        
        # For one-shot, directly from config_details
        if 'config_details' in log_data and 'target_filter_pruning_rate_per_layer' in log_data['config_details']:
            rate = log_data['config_details']['target_filter_pruning_rate_per_layer']
            # print(f"    One-shot rate from {log_file_path}: {rate}") # Verbose
            return {'type': 'one-shot', 'rate': float(rate)}
        
        # For a single iterative stage, get its own applied rate
        if 'config_details' in log_data and 'applied_step_rate_for_this_stage' in log_data['config_details']:
            rate = log_data['config_details']['applied_step_rate_for_this_stage']
            # print(f"    Iterative step rate from {log_file_path}: {rate}") # Verbose
            return {'type': 'iterative_step', 'rate': float(rate)} # Special type for accumulation

    except json.JSONDecodeError:
        print(f"    Error decoding JSON from {log_file_path}")
    except Exception as e:
        print(f"    Error processing log {log_file_path}: {e}")
    return None

if __name__ == "__main__":
    if not os.path.exists(VALIDATION_DATA_PATH) or not os.listdir(VALIDATION_DATA_PATH):
        print(f"FATAL ERROR: VALIDATION_DATA_PATH '{VALIDATION_DATA_PATH}' is missing, empty or not readable."); exit()

    all_results_list = []
    
    # --- Scan for experiment directories ---
    experiment_base_dirs = []
    for dir_list_category in [ONE_SHOT_STRUCTURED_MODEL_DIRS, ITERATIVE_STRUCTURED_MODEL_DIRS]:
        for path_pattern_or_dir in dir_list_category:
            # This logic assumes path_pattern_or_dir is a category like "pruning_structured_oneshot"
            # And actual experiments are subfolders like "resnet50_prune_struct_os_l1filter_fp30_ft"
            for exp_dir_globbed in glob.glob(os.path.join(path_pattern_or_dir, "*")):
                if os.path.isdir(exp_dir_globbed):
                    if exp_dir_globbed not in experiment_base_dirs:
                         experiment_base_dirs.append(exp_dir_globbed)
    
    experiment_dirs_to_scan = sorted(list(set(experiment_base_dirs)))
    print(f"Found {len(experiment_dirs_to_scan)} potential experiment directories to scan:")
    for d in experiment_dirs_to_scan: print(f"  - {d}")

    # --- Process One-Shot Experiments ---
    print("\n--- Processing One-Shot Structured Pruning Experiments ---")
    for exp_path_str in experiment_dirs_to_scan:
        exp_path = Path(exp_path_str)
        if not (("oneshot" in exp_path.name.lower() or "os" in exp_path.name.lower()) and \
                any(parent.name.lower() == "pruning_structured_oneshot" for parent in exp_path.parents)):
            continue # Skip if not identified as a one-shot experiment

        print(f"\nProcessing One-Shot experiment directory: {exp_path_str}")
        model_file_path = exp_path / "model_final.pth" # Your logs save as 'model_final.pth' it seems
        log_file_path = exp_path / "log.json"

        if not model_file_path.exists():
            print(f"  Model file 'model_final.pth' not found in {exp_path_str}. Skipping.")
            continue
        
        exp_folder_name = exp_path.name
        model_file_stem = model_file_path.stem
        experiment_id = f"{exp_folder_name}_{model_file_stem}"
        print(f"  Found model file: {model_file_path.name}")

        pruning_rec_config = get_pruning_config_from_log(log_file_path)

        if pruning_rec_config and pruning_rec_config.get('type') == 'one-shot':
            print(f"    Reconstruction config for {model_file_path.name}: {pruning_rec_config}")
            model_metrics = process_single_structured_pruned_model(str(model_file_path), experiment_id, pruning_rec_config)
            all_results_list.append(model_metrics)
            print(f"    Finished processing {model_file_path.name}")
        else:
            print(f"    SKIPPING {model_file_path.name}: Could not determine one-shot pruning config from {log_file_path}.")
            all_results_list.append({'Experiment_ID': experiment_id, 'Model_Path': str(model_file_path),
                                     'Final_Val_Accuracy': "N/A (Pruning Config Error)", 'Load_Status': "Skipped - Config Error",
                                     'Num_Classes': FIXED_NUM_CLASSES})
        gc.collect()

    # --- Process Iterative Experiments (requires ordered processing of stages) ---
    print("\n--- Processing Iterative Structured Pruning Experiments ---")
    # We need to group iterative experiments by their base name and sort by stage
    iterative_exp_groups = {}
    for exp_path_str in experiment_dirs_to_scan:
        exp_path = Path(exp_path_str)
        if not (("iterative" in exp_path.name.lower() or "it" in exp_path.name.lower()) and \
                 any(parent.name.lower() == "pruning_structured_iterative" for parent in exp_path.parents)):
            continue

        match = re.search(r"(stage(\d+))", exp_path.name.lower())
        if match:
            stage_num = int(match.group(2))
            # Infer a base name, e.g., "resnet50_prune_struct_it_l1filter"
            base_name = exp_path.name.lower().split(match.group(1))[0].rstrip('_') 
            if base_name not in iterative_exp_groups:
                iterative_exp_groups[base_name] = []
            iterative_exp_groups[base_name].append({'path': exp_path_str, 'stage': stage_num, 'name': exp_path.name})

    for base_name, stages_info in iterative_exp_groups.items():
        print(f"\nProcessing Iterative Group: {base_name}")
        sorted_stages = sorted(stages_info, key=lambda x: x['stage'])
        
        cumulative_step_rates = []
        for stage_info in sorted_stages:
            exp_path_str = stage_info['path']
            exp_path = Path(exp_path_str)
            current_stage_num = stage_info['stage']
            
            print(f"  Processing Stage {current_stage_num}: {exp_path_str}")
            model_file_path = exp_path / "model_final.pth"
            log_file_path = exp_path / "log.json"

            if not model_file_path.exists():
                print(f"    Model file 'model_final.pth' not found in {exp_path_str}. Skipping this stage.")
                continue

            exp_folder_name = exp_path.name
            model_file_stem = model_file_path.stem
            experiment_id = f"{exp_folder_name}_{model_file_stem}"
            print(f"    Found model file: {model_file_path.name}")

            # Get the rate for THIS specific stage
            current_stage_pruning_info = get_pruning_config_from_log(log_file_path)
            
            if current_stage_pruning_info and current_stage_pruning_info.get('type') == 'iterative_step':
                cumulative_step_rates.append(current_stage_pruning_info['rate'])
                # Now, the pruning_rec_config for *this* model is the cumulative list of rates
                pruning_rec_config = {'type': 'iterative', 'step_rates': list(cumulative_step_rates)} # Use a copy
                
                print(f"      Reconstruction config for Stage {current_stage_num} model ({model_file_path.name}): {pruning_rec_config}")
                model_metrics = process_single_structured_pruned_model(str(model_file_path), experiment_id, pruning_rec_config)
                all_results_list.append(model_metrics)
                print(f"      Finished processing Stage {current_stage_num} model.")
            else:
                print(f"      SKIPPING Stage {current_stage_num} model ({model_file_path.name}): Could not get iterative step rate from {log_file_path}.")
                all_results_list.append({'Experiment_ID': experiment_id, 'Model_Path': str(model_file_path),
                                         'Final_Val_Accuracy': "N/A (Pruning Config Error for Stage)", 
                                         'Load_Status': "Skipped - Stage Config Error", 'Num_Classes': FIXED_NUM_CLASSES})
            gc.collect()

    # --- Create DataFrame and Save ---
    if not all_results_list:
        print("\nNo models were processed. Output CSV will be empty or not created.")
    else:
        df_results = pd.DataFrame(all_results_list)
        desired_cols = [
            'Experiment_ID', 'Model_Path', 'Num_Classes', 'Pruning_Type', 
            'Pruning_Params_Detail', 'Load_Status', 'Final_Val_Accuracy', 
            'Model_Size_MB_Disk', 'Params_Millions', 'FLOPs_GMACs',
            'Inference_Time_ms_CPU (Batch 1)', 'Inference_Time_ms_GPU (Batch 1)'
        ]
        for col in df_results.columns:
            if col not in desired_cols: desired_cols.append(col)
        df_results = df_results.reindex(columns=desired_cols)
        df_results.to_csv(OUTPUT_CSV, index=False, lineterminator='\n', float_format='%.5f')
        print(f"\n--- Evaluation summary for structured pruning saved to {OUTPUT_CSV} ---")
        if not df_results.empty:
            print("\nFirst 5 rows of the summary:")
            print(df_results.head().to_string())
        else: print("DataFrame is empty.")

    print(f"Total models attempted: {len(all_results_list)}")
    if all_results_list:
        successful_loads = sum(1 for r in all_results_list if r.get('Load_Status') == "Success")
        print(f"Successfully loaded and processed: {successful_loads}")
    print("--- Standalone Script Finished ---")