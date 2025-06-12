import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
# import torchvision.transforms as transforms # Not strictly needed if only for INPUT_TENSOR_CPU
import time
from pathlib import Path
import glob
import traceback
import gc
from thop import profile
import torch_pruning as tp
import re
import torch.backends.quantized # For quantized engine

print("--- Metrics Generator Script Starting (v3 - Full) ---")

# --- Configuration ---
ROOT_DIR = "saved_models_and_logs"
METRICS_OUTPUT_CSV = "remeasured_critical_metrics_v3.csv"
DEFAULT_NUM_CLASSES = 1000
FIXED_NUM_CLASSES = 1000 # For reconstruction consistency

INPUT_TENSOR_CPU = torch.randn(1, 3, 224, 224)
INPUT_TENSOR_GPU = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    try:
        INPUT_TENSOR_GPU = INPUT_TENSOR_CPU.to(DEVICE)
    except Exception as e_cuda_init:
        print(f"ERROR initializing INPUT_TENSOR_GPU on CUDA: {e_cuda_init}")

WARMUP_INFERENCES = 5
TIMED_INFERENCES = 10

GPU_UNSTABLE_QUANTIZED_MODELS = [
    "resnet18pretrained_distilled_quant_ptq_int8_perchannel_post",
    "resnet18pretrained_distilled_quant_ptq_int8_pertensor_post",
    "resnet18pretrained_distilled_quant_qat_int8_epochs8",
    "resnet50_quant_ptq_int8_perchannel_post",
    "resnet50_quant_ptq_int8_pertensor_post",
    "resnet50_quant_qat_int8_epochs8",
]

# --- Helper Functions ---
def get_model_file_path(experiment_path_str):
    experiment_path = Path(experiment_path_str)
    pth_files = list(experiment_path.glob("*.pth"))
    if not pth_files:
        return None

    # Prioritize common names
    for common_name in ["model_final.pth", "model_quantized.pth", "model_scripted.pth"]:
        for p_file in pth_files:
            if p_file.name == common_name:
                return str(p_file)
    
    # Prioritize specific baseline names if relevant
    for p_file in pth_files:
        if "baseline_ft_imagenetmini_final.pth" in p_file.name: # Example baseline name
            return str(p_file)
            
    # Fallback: most recently modified .pth file
    try:
        pth_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if pth_files:
            return str(pth_files[0])
    except Exception: # os.path.getmtime might fail if file deleted during glob
        if pth_files: # If sort failed but list not empty, return first found
             return str(pth_files[0])
    return None


def get_base_resnet_model_for_reconstruction(base_arch_name="ResNet50"): # Parameterize base arch
    if base_arch_name == "ResNet50":
        model = models.resnet50(weights=None, num_classes=FIXED_NUM_CLASSES)
    elif base_arch_name == "ResNet18":
        model = models.resnet18(weights=None, num_classes=FIXED_NUM_CLASSES)
    else:
        print(f"ERROR: Base architecture '{base_arch_name}' not supported for reconstruction.")
        return None
    return model

def apply_structured_pruning_to_model_for_reconstruction(model_to_prune, example_inputs, target_pruning_rate, dev):
    model_to_prune.to(dev)
    example_inputs = example_inputs.to(dev)
    ignored_layers = []
    # Correctly find Linear layers for ResNet family
    for name, m in model_to_prune.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == FIXED_NUM_CLASSES:
            # Check if it's the final classifier layer
            if name == 'fc' or name.endswith('.fc'): # Common names for ResNet classifier
                ignored_layers.append(m)
                # print(f"    Ignoring layer for pruning reconstruction: {name}")

    importance = tp.importance.MagnitudeImportance(p=1) # L1
    pruner = tp.pruner.MagnitudePruner(
        model=model_to_prune, example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=target_pruning_rate, global_pruning=False, # Usually layer-wise for this rate
        ignored_layers=ignored_layers
    )
    pruner.step()
    return model_to_prune

def get_pruning_config_from_log_for_reconstruction(log_file_path):
    if not log_file_path.exists(): return None
    try:
        with open(log_file_path, 'r') as f: log_data = json.load(f)
        if 'config_details' in log_data:
            cd = log_data['config_details']
            if 'target_filter_pruning_rate_per_layer' in cd:
                return {'type': 'one-shot', 'rate': float(cd['target_filter_pruning_rate_per_layer'])}
            if 'applied_step_rate_for_this_stage' in cd: # Key for iterative
                return {'type': 'iterative_step', 'rate': float(cd['applied_step_rate_for_this_stage'])}
    except Exception as e: print(f"    Log parse error for {log_file_path.name}: {e}")
    return None

iterative_pruning_configs_cache = {}

def build_iterative_pruning_config(experiment_dir_path_str, all_experiment_dirs_in_category):
    exp_path_obj = Path(experiment_dir_path_str)
    exp_name_lower = exp_path_obj.name.lower()
    
    match = re.search(r"(stage(\d+))", exp_name_lower)
    if not match: return None # Not an iterative stage identifiable by "stageX"

    current_stage_num = int(match.group(2))
    # More robust base name extraction: remove "_stageX" and any *immediately preceding* underscore
    base_name_for_iter = re.sub(r"_?stage\d+.*$", "", exp_name_lower).rstrip('_')
    if not base_name_for_iter: # Edge case if name was just "stageX_..."
        base_name_for_iter = exp_name_lower.split(match.group(1))[0].rstrip('_')


    # Check cache first
    if base_name_for_iter in iterative_pruning_configs_cache and \
       current_stage_num in iterative_pruning_configs_cache[base_name_for_iter]:
        return iterative_pruning_configs_cache[base_name_for_iter][current_stage_num]

    sibling_stages = []
    for potential_sibling_path_str in all_experiment_dirs_in_category:
        potential_sibling_path = Path(potential_sibling_path_str)
        # Check if base_name_for_iter is a prefix of the potential sibling's name
        # and that it contains "stage" to avoid matching unrelated experiments
        if potential_sibling_path.name.lower().startswith(base_name_for_iter) and \
           "stage" in potential_sibling_path.name.lower():
            sibling_match = re.search(r"stage(\d+)", potential_sibling_path.name.lower())
            if sibling_match:
                sibling_stage_num = int(sibling_match.group(1))
                sibling_stages.append({'path': potential_sibling_path, 'stage': sibling_stage_num, 'name': potential_sibling_path.name})
    
    if not sibling_stages: # No stages found for this base name
        # print(f"    DEBUG: No sibling stages found for base '{base_name_for_iter}' from {exp_name_lower}")
        return None

    sorted_stages = sorted(sibling_stages, key=lambda x: x['stage'])
    
    cumulative_step_rates = []
    final_config_for_current_stage = None

    if base_name_for_iter not in iterative_pruning_configs_cache:
        iterative_pruning_configs_cache[base_name_for_iter] = {}

    for stage_info in sorted_stages:
        if stage_info['stage'] > current_stage_num: 
            break # Only process up to and including the current stage
        stage_log_path = stage_info['path'] / "log.json"
        stage_pruning_info = get_pruning_config_from_log_for_reconstruction(stage_log_path)
        
        if stage_pruning_info and stage_pruning_info.get('type') == 'iterative_step':
            cumulative_step_rates.append(stage_pruning_info['rate'])
            current_stage_config = {'type': 'iterative', 'step_rates': list(cumulative_step_rates)} # Use a copy
            iterative_pruning_configs_cache[base_name_for_iter][stage_info['stage']] = current_stage_config
            if stage_info['stage'] == current_stage_num:
                final_config_for_current_stage = current_stage_config
        else:
            print(f"    ERROR: Could not get iterative_step rate for {stage_info['name']} (log: {stage_log_path.name}) in group {base_name_for_iter}. Halting accumulation for this group.")
            if base_name_for_iter in iterative_pruning_configs_cache: # Invalidate if any step fails
                del iterative_pruning_configs_cache[base_name_for_iter]
            return None # Cannot reliably build cumulative config if a step is missing/bad
            
    return final_config_for_current_stage


def load_and_reconstruct_structured_pruned_model(model_path_str, experiment_dir_path_str, base_arch_for_recon, dev, all_exp_dirs_in_cat):
    log_file_path = Path(experiment_dir_path_str) / "log.json"
    pruning_config_for_this_model = None

    # Try to get one-shot config first from the specific log
    one_shot_config = get_pruning_config_from_log_for_reconstruction(log_file_path)
    if one_shot_config and one_shot_config['type'] == 'one-shot':
        pruning_config_for_this_model = one_shot_config
    # Check if experiment_dir_path_str suggests iterative, or if the log explicitly says 'iterative_step'
    elif ("iterative" in Path(experiment_dir_path_str).name.lower() or \
          "it" in Path(experiment_dir_path_str).name.lower() or \
          (one_shot_config and one_shot_config['type'] == 'iterative_step')): # If log for this stage says it's an iter_step
        if all_exp_dirs_in_cat is None:
            print(f"    ERROR: Iterative model {Path(experiment_dir_path_str).name} but context for sibling experiment dirs not provided.")
            return None
        # print(f"    DEBUG: Building iterative config for {Path(experiment_dir_path_str).name}")
        pruning_config_for_this_model = build_iterative_pruning_config(experiment_dir_path_str, all_exp_dirs_in_cat)
    
    if not pruning_config_for_this_model:
        # This means it's not a torch-pruning one-shot or a recognized iterative step
        # print(f"    DEBUG: No torch-pruning one-shot or iterative config determined for {Path(experiment_dir_path_str).name} from its log.")
        return None 

    reconstructed_model = get_base_resnet_model_for_reconstruction(base_arch_for_recon)
    if reconstructed_model is None: return None # Base model not supported
    reconstructed_model.to(dev)
    example_inputs_local = INPUT_TENSOR_CPU.to(dev) # Assuming INPUT_TENSOR_CPU is globally defined

    try:
        if pruning_config_for_this_model['type'] == 'one-shot':
            reconstructed_model = apply_structured_pruning_to_model_for_reconstruction(
                reconstructed_model, example_inputs_local, pruning_config_for_this_model['rate'], dev)
        elif pruning_config_for_this_model['type'] == 'iterative':
            # print(f"    Reconstructing iterative with steps: {pruning_config_for_this_model['step_rates']}")
            current_arch_model = reconstructed_model # Start with the base model for this arch
            for i, step_rate in enumerate(pruning_config_for_this_model['step_rates']):
                # print(f"      Applying iter step {i+1}, rate {step_rate}")
                current_arch_model = apply_structured_pruning_to_model_for_reconstruction(
                    current_arch_model, example_inputs_local, step_rate, dev)
            reconstructed_model = current_arch_model
        else:
            print(f"    ERROR: Unknown final pruning_config type for reconstruction: {pruning_config_for_this_model['type']}")
            return None

        # Load state_dict with weights_only=False as a safe default, can be changed to True if only loading weights.
        # For models saved by torch-pruning examples, they are usually just state_dicts.
        state_dict = torch.load(model_path_str, map_location=dev, weights_only=True) # Try True first
        # Clean state_dict keys
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # Handle nested state_dicts if model was saved inside a checkpoint dictionary
        if 'model' in state_dict and isinstance(state_dict['model'], dict): 
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict): 
            state_dict = state_dict['state_dict']

        reconstructed_model.load_state_dict(state_dict)
        # print(f"    State_dict loaded successfully into RECONSTRUCTED torch-pruning model: {Path(model_path_str).name}")
        reconstructed_model.eval()
        return reconstructed_model
    except RuntimeError as e_sd_load: # Catch specific state_dict loading errors
        if "weights_only" in str(e_sd_load) and "cannot be loaded with weights_only=True" in str(e_sd_load):
            print(f"    INFO: Loading {Path(model_path_str).name} with weights_only=True failed, retrying with weights_only=False.")
            try: # Retry with weights_only=False
                state_dict = torch.load(model_path_str, map_location=dev, weights_only=False)
                if all(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                if 'model' in state_dict and isinstance(state_dict['model'], dict): state_dict = state_dict['model']
                elif 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict): state_dict = state_dict['state_dict']
                reconstructed_model.load_state_dict(state_dict)
                reconstructed_model.eval()
                return reconstructed_model
            except Exception as e_retry:
                 print(f"    ERROR loading/reconstructing structured pruned model {Path(experiment_dir_path_str).name} (retry failed): {e_retry}")
                 return None
        else:
            print(f"    ERROR loading/reconstructing structured pruned model {Path(experiment_dir_path_str).name}: {e_sd_load}")
            return None
    except Exception as e: # Catch other errors
        print(f"    ERROR loading/reconstructing structured pruned model {Path(experiment_dir_path_str).name}: {e}")
        # traceback.print_exc()
        return None

def measure_critical_metrics_for_model(
    model_file_to_load,
    experiment_dir_path_str,
    base_arch,
    num_classes,
    experiment_id,
    all_exp_dirs_in_cat, # For iterative structured pruning context
    baseline_metrics_dict # For FLOPs/Params fallback
    ):
    
    metrics = {
        'FLOPs_GMACs': pd.NA,
        'Params_Millions': pd.NA,
        'Inference_Time_ms_CPU (Batch 1)': pd.NA,
        'Inference_Time_ms_GPU (Batch 1)': pd.NA,
    }
    if not model_file_to_load or not os.path.exists(model_file_to_load):
        print(f"      ERROR ({experiment_id}): Model file not found at {model_file_to_load}")
        return metrics

    exp_id_lower = experiment_id.lower()
    # Use folder name for type detection for more consistency with how logs might be structured
    exp_folder_name_lower = Path(experiment_dir_path_str).name.lower()

    # Detect if it's a torch-pruning structured model (needs reconstruction)
    is_tp_structured = "prune_struct_it" in exp_folder_name_lower or \
                       "prune_struct_os" in exp_folder_name_lower or \
                       "pruning_structured_iterative" in exp_folder_name_lower or \
                       "pruning_structured_oneshot" in exp_folder_name_lower
    
    is_quantized_ao = "quant_ptq" in exp_folder_name_lower or \
                      "quant_qat" in exp_folder_name_lower or \
                      "ptq_int8" in exp_folder_name_lower or \
                      "qat_int8" in exp_folder_name_lower
                      # "quantization_ptq" in exp_folder_name_lower or \ # if category name also used
                      # "quantization_qat" in exp_folder_name_lower
    
    is_kmeans_quantized = "quant_kmeans" in exp_folder_name_lower # or "quantization_kmeans" in exp_folder_name_lower


    model_for_measurement = None
    model_loaded_successfully = False
    
    original_quantized_engine = None # To restore if changed
    if is_quantized_ao: # Only set for AO quantization for now
        try:
            if hasattr(torch.backends.quantized, 'engine'):
                original_quantized_engine = torch.backends.quantized.engine
                # Try 'x86' for broader compatibility on x86 CPUs,
                # or 'fbgemm' if you know it was used. 'qnnpack' for ARM.
                torch.backends.quantized.engine = 'x86' 
                print(f"      INFO ({experiment_id}): Attempting to set quantized engine to 'x86'.")
        except Exception as e_engine_set:
            print(f"      WARNING ({experiment_id}): Could not set/get quantized engine: {e_engine_set}")

    if is_tp_structured:
        # print(f"      INFO ({experiment_id}): torch-pruning structured. Attempting reconstruction.")
        model_for_measurement = load_and_reconstruct_structured_pruned_model(
            model_file_to_load, experiment_dir_path_str, base_arch, DEVICE, all_exp_dirs_in_cat
        )
        if model_for_measurement: model_loaded_successfully = True
    
    if not model_loaded_successfully: # Fallback or non-tp-structured
        try:
            # print(f"      Attempting JIT load for {experiment_id}")
            model_for_measurement = torch.jit.load(model_file_to_load, map_location=DEVICE)
            model_loaded_successfully = True
        except Exception as e_jit:
            # print(f"      DEBUG ({experiment_id}): JIT load failed: {e_jit}")
            try:
                # print(f"      Attempting torch.load (full model or state_dict) for {experiment_id}")
                # Try with weights_only=True first for state_dicts, then False for full models
                loaded_content = None
                try:
                    loaded_content = torch.load(model_file_to_load, map_location=DEVICE, weights_only=True)
                except RuntimeError as e_wo_true: # If weights_only=True fails (e.g. loading pickled object)
                    if "cannot be loaded with weights_only=True" in str(e_wo_true):
                        # print(f"        weights_only=True failed for {experiment_id}, retrying with False.")
                        loaded_content = torch.load(model_file_to_load, map_location=DEVICE, weights_only=False)
                    else: raise e_wo_true # Re-raise other RuntimeErrors
                except Exception as e_other_load: # Catch other load errors
                    raise e_other_load


                if isinstance(loaded_content, torch.nn.Module):
                    model_for_measurement = loaded_content
                    model_loaded_successfully = True
                elif isinstance(loaded_content, dict): # Assumed to be a state_dict
                    # print(f"        Loaded state_dict for {experiment_id}. Reconstructing base model.")
                    instance = get_base_resnet_model_for_reconstruction(base_arch)
                    if instance is None: return metrics # Base arch not supported
                    
                    state_dict_to_load = loaded_content
                    if any(k.startswith('module.') for k in state_dict_to_load.keys()):
                        state_dict_to_load = {k.replace('module.', ''): v for k, v in state_dict_to_load.items()}
                    if 'model' in state_dict_to_load and isinstance(state_dict_to_load['model'], dict):
                        state_dict_to_load = state_dict_to_load['model']
                    elif 'state_dict' in state_dict_to_load and isinstance(state_dict_to_load['state_dict'], dict):
                        state_dict_to_load = state_dict_to_load['state_dict']
                    
                    instance.load_state_dict(state_dict_to_load)
                    model_for_measurement = instance.to(DEVICE)
                    model_loaded_successfully = True
                else:
                    print(f"      ERROR ({experiment_id}): Loaded object via torch.load is not nn.Module or dict: {type(loaded_content)}")
            except Exception as e_load:
                print(f"      ERROR ({experiment_id}): General torch.load also failed: {e_load}")

    if original_quantized_engine and hasattr(torch.backends.quantized, 'engine'): # Restore engine
        try:
            torch.backends.quantized.engine = original_quantized_engine
            # print(f"      INFO ({experiment_id}): Restored quantized engine to {original_quantized_engine}.")
        except Exception: pass


    if not model_loaded_successfully or model_for_measurement is None:
        print(f"      FINAL ERROR ({experiment_id}): Could not prepare any model for measurement.")
        return metrics

    model_for_measurement.eval()

    # --- FLOPs and Params ---
    use_baseline_flops_params_for_this_model = False
    if is_quantized_ao or is_kmeans_quantized: # If it's any kind of quantization we know
        use_baseline_flops_params_for_this_model = True
    
    if use_baseline_flops_params_for_this_model:
        if base_arch in baseline_metrics_dict and baseline_metrics_dict[base_arch]:
            metrics['FLOPs_GMACs'] = baseline_metrics_dict[base_arch].get("flops_gmacs_raw", pd.NA)
            metrics['Params_Millions'] = baseline_metrics_dict[base_arch].get("params_millions_raw", pd.NA)
        else:
            print(f"      WARNING ({experiment_id}): Baseline FLOPs/Params for {base_arch} not found for quantized model.")
    else: # Attempt thop for non-quantized
        thop_success = False
        if isinstance(model_for_measurement, torch.nn.Module) and not isinstance(model_for_measurement, torch.jit.ScriptModule):
            try:
                model_for_thop = model_for_measurement.to('cpu')
                macs, params = profile(model_for_thop, inputs=(INPUT_TENSOR_CPU.to('cpu'),), verbose=False)
                metrics['FLOPs_GMACs'] = macs / 1e9
                metrics['Params_Millions'] = params / 1e6
                thop_success = True
                del model_for_thop
            except Exception as e_flops:
                print(f"      WARNING ({experiment_id}): thop profiling failed: {e_flops}")
        
        if not thop_success:
            if base_arch in baseline_metrics_dict and baseline_metrics_dict[base_arch]:
                metrics['FLOPs_GMACs'] = baseline_metrics_dict[base_arch].get("flops_gmacs_raw", pd.NA)
                metrics['Params_Millions'] = baseline_metrics_dict[base_arch].get("params_millions_raw", pd.NA)
            else: print(f"      WARNING ({experiment_id}): Thop failed AND baseline FLOPs/Params for {base_arch} not found.")

    # --- Inference Speed ---
    try:
        model_cpu_timing = model_for_measurement.to('cpu')
        with torch.no_grad():
            for _ in range(WARMUP_INFERENCES): _ = model_cpu_timing(INPUT_TENSOR_CPU)
            timings_cpu = []
            for _ in range(TIMED_INFERENCES):
                start_time = time.perf_counter(); _ = model_cpu_timing(INPUT_TENSOR_CPU); end_time = time.perf_counter()
                timings_cpu.append((end_time - start_time) * 1000)
            metrics['Inference_Time_ms_CPU (Batch 1)'] = sum(timings_cpu) / len(timings_cpu) if timings_cpu else pd.NA
        del model_cpu_timing
    except Exception as e_cpu_time: print(f"      ERROR ({experiment_id}): CPU timing failed: {e_cpu_time}")

    if DEVICE.type == 'cuda' and INPUT_TENSOR_GPU is not None:
        if experiment_id in GPU_UNSTABLE_QUANTIZED_MODELS:
            metrics['Inference_Time_ms_GPU (Batch 1)'] = "N/A (Known JIT GPU Unstable)"
        else:
            try:
                model_gpu_timing = model_for_measurement.to(DEVICE)
                with torch.no_grad():
                    for _ in range(WARMUP_INFERENCES): _ = model_gpu_timing(INPUT_TENSOR_GPU); torch.cuda.synchronize()
                    timings_gpu = []
                    for _ in range(TIMED_INFERENCES):
                        torch.cuda.synchronize(); t0 = time.perf_counter(); _ = model_gpu_timing(INPUT_TENSOR_GPU); torch.cuda.synchronize(); t1 = time.perf_counter()
                        timings_gpu.append((t1 - t0) * 1000)
                    metrics['Inference_Time_ms_GPU (Batch 1)'] = sum(timings_gpu) / len(timings_gpu) if timings_gpu else pd.NA
                if model_gpu_timing is not model_for_measurement: del model_gpu_timing
            except Exception as e_gpu_time: print(f"      ERROR ({experiment_id}): GPU timing failed: {e_gpu_time}")
    
    if model_for_measurement: del model_for_measurement
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    return metrics

# --- Main ---
if __name__ == "__main__":
    all_results_list = []
    baseline_metrics_for_fallback = {}
    print("--- Initializing... ---")

    all_experiment_dirs_by_category = {}
    if not os.path.exists(ROOT_DIR):
        print(f"FATAL: ROOT_DIR '{ROOT_DIR}' does not exist.")
        exit()
        
    for cat_name_scan in os.listdir(ROOT_DIR):
        cat_path_scan = os.path.join(ROOT_DIR, cat_name_scan)
        if not os.path.isdir(cat_path_scan): continue
        all_experiment_dirs_by_category[cat_name_scan] = []
        for exp_name_scan in os.listdir(cat_path_scan):
            exp_path_scan = os.path.join(cat_path_scan, exp_name_scan)
            if os.path.isdir(exp_path_scan):
                all_experiment_dirs_by_category[cat_name_scan].append(exp_path_scan)
    
    iterative_pruning_configs_cache.clear()

    print("--- Pass 1: Processing Baselines to establish FLOPs/Params references ---")
    for cat_name_key in list(all_experiment_dirs_by_category.keys()):
        for exp_dir_path_str in all_experiment_dirs_by_category[cat_name_key]:
            exp_name = Path(exp_dir_path_str).name
            if "baseline" in exp_name.lower(): # Identify baselines by name
                print(f"\nProcessing BASELINE experiment: {exp_name} from category '{cat_name_key}'")
                model_file = get_model_file_path(exp_dir_path_str)
                if not model_file:
                    print(f"  SKIPPING Baseline {exp_name}: Model .pth file not found.")
                    all_results_list.append({'Experiment_ID': exp_name, 'FLOPs_GMACs': pd.NA, 'Params_Millions': pd.NA, 'Inference_Time_ms_CPU (Batch 1)': pd.NA, 'Inference_Time_ms_GPU (Batch 1)': pd.NA})
                    continue

                base_arch_guess = "Unknown"
                if "resnet18" in exp_name.lower(): base_arch_guess = "ResNet18"
                elif "resnet50" in exp_name.lower(): base_arch_guess = "ResNet50"

                num_classes_from_log = DEFAULT_NUM_CLASSES
                # Optional: Parse num_classes from baseline's log if available and needed
                # log_p_baseline = Path(exp_dir_path_str) / "log.json" ...

                metrics_data = measure_critical_metrics_for_model(
                    str(model_file), exp_dir_path_str, base_arch_guess, num_classes_from_log,
                    exp_name, all_experiment_dirs_by_category.get(cat_name_key), {} # Pass empty dict for baselines
                )
                all_results_list.append({'Experiment_ID': exp_name, **metrics_data})

                if base_arch_guess != "Unknown":
                    flops_val, params_val = metrics_data.get('FLOPs_GMACs'), metrics_data.get('Params_Millions')
                    if pd.notna(flops_val) and pd.notna(params_val):
                        baseline_metrics_for_fallback[base_arch_guess] = {
                            "flops_gmacs_raw": flops_val, "params_millions_raw": params_val
                        }
                        print(f"  Stored baseline reference for {base_arch_guess}: FLOPs={flops_val:.2f}G, Params={params_val:.2f}M")
                    else: print(f"  WARNING: FLOPs/Params for baseline {exp_name} (arch {base_arch_guess}) are NA. Cannot store as reference.")
    
    print(f"\n--- Baseline references populated: {baseline_metrics_for_fallback} ---")
    
    print("\n--- Pass 2: Processing ALL other (non-baseline) experiments ---")
    for cat_name_key in list(all_experiment_dirs_by_category.keys()):
        for exp_dir_path_str in all_experiment_dirs_by_category[cat_name_key]:
            exp_name = Path(exp_dir_path_str).name
            if "baseline" in exp_name.lower(): continue # Skip baselines, already processed

            print(f"\n  Processing experiment: {exp_name} from category '{cat_name_key}'")
            model_file = get_model_file_path(exp_dir_path_str)
            if not model_file:
                print(f"    SKIPPING {exp_name}: Model .pth file not found.")
                all_results_list.append({'Experiment_ID': exp_name, 'FLOPs_GMACs': pd.NA, 'Params_Millions': pd.NA, 'Inference_Time_ms_CPU (Batch 1)': pd.NA, 'Inference_Time_ms_GPU (Batch 1)': pd.NA})
                continue

            base_arch_guess = "ResNet50" # Default, adjust as per your naming/conventions
            if "resnet18" in exp_name.lower(): base_arch_guess = "ResNet18"
            
            num_classes_from_log = DEFAULT_NUM_CLASSES
            log_p = Path(exp_dir_path_str) / "log.json"
            if log_p.exists():
                try:
                    with open(log_p, 'r') as f_log: log_data_temp = json.load(f_log)
                    cfg_details = log_data_temp.get('config_details', {})
                    num_classes_from_log = cfg_details.get('num_classes', DEFAULT_NUM_CLASSES)
                    if 'student_config' in cfg_details and isinstance(cfg_details['student_config'], dict):
                        num_classes_from_log = cfg_details['student_config'].get('num_classes', num_classes_from_log)
                except Exception as e_log: print(f"    Warning: Could not parse num_classes from log for {exp_name}: {e_log}")

            metrics_data = measure_critical_metrics_for_model(
                str(model_file), exp_dir_path_str, base_arch_guess, num_classes_from_log,
                exp_name, all_experiment_dirs_by_category.get(cat_name_key), baseline_metrics_for_fallback
            )
            all_results_list.append({'Experiment_ID': exp_name, **metrics_data})

    if not all_results_list:
        print("\nNo models were processed. Output CSV will be empty.")
    else:
        # Deduplicate results by Experiment_ID, keeping the last entry (from Pass 2 for non-baselines)
        # Baselines from Pass 1 will be included if not overridden by a non-baseline with the same name.
        # This handles if an experiment was accidentally named "baseline_something_else"
        temp_df = pd.DataFrame(all_results_list)
        if not temp_df.empty:
            df_results = temp_df.drop_duplicates(subset=['Experiment_ID'], keep='last')
        else:
            df_results = temp_df

        cols = ['Experiment_ID', 'FLOPs_GMACs', 'Params_Millions', 
                'Inference_Time_ms_CPU (Batch 1)', 'Inference_Time_ms_GPU (Batch 1)']
        for c in cols: # Ensure columns exist
            if c not in df_results.columns: df_results[c] = pd.NA
        df_results = df_results.reindex(columns=cols)
        
        df_results.to_csv(METRICS_OUTPUT_CSV, index=False, lineterminator='\n', float_format='%.5f')
        print(f"\n--- Remeasured critical metrics saved to {METRICS_OUTPUT_CSV} ---")
        if not df_results.empty:
            print("\nFirst 5 rows of the remeasured metrics:")
            print(df_results.head().to_string())

    print("--- Metrics Generator Script Finished ---")