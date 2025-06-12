import os
import json
import pandas as pd
import torch
import torch.nn as nn # Added for nn.Linear check
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import time
from pathlib import Path
import glob
import traceback
import gc # For garbage collection
from thop import profile # Import thop
import torch_pruning as tp # <-- ADDED for structured pruning reconstruction
import re # <-- ADDED for parsing stage numbers (if needed for iterative logs)

print("--- Hybrid Script Starting: Imports completed ---")

# --- Configuration (from Script 2, with additions from Script 1 if necessary) ---
ROOT_DIR = "saved_models_and_logs"
OUTPUT_CSV = "model_optimization_summary_hybrid.csv" # New output file
DEFAULT_NUM_CLASSES = 1000
FIXED_NUM_CLASSES = 1000 # From Script 1, ensure consistency

# --- Uniform Evaluation Configuration (from Script 2) ---
VALIDATION_DATA_PATH = "imagenet-mini/val"
BATCH_SIZE_EVAL = 32
NUM_WORKERS_EVAL = 0
MAX_EVAL_BATCHES = 125

INPUT_TENSOR_CPU = torch.randn(1, 3, 224, 224)
INPUT_TENSOR_GPU = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define DEVICE earlier

if torch.cuda.is_available():
    try:
        INPUT_TENSOR_GPU = INPUT_TENSOR_CPU.to(DEVICE) # Use DEVICE
    except Exception as e_cuda_init:
        print(f"ERROR initializing INPUT_TENSOR_GPU on CUDA: {e_cuda_init}")

WARMUP_INFERENCES = 2 # Script 2's value
TIMED_INFERENCES = 5  # Script 2's value

GPU_UNSTABLE_QUANTIZED_MODELS = [
    "resnet18pretrained_distilled_quant_ptq_int8_perchannel_post",
    "resnet18pretrained_distilled_quant_ptq_int8_pertensor_post",
    "resnet18pretrained_distilled_quant_qat_int8_epochs8",
    "resnet50_quant_ptq_int8_perchannel_post",
    "resnet50_quant_ptq_int8_pertensor_post",
    "resnet50_quant_qat_int8_epochs8",
]

# --- Helper: Image Transforms (from Script 2) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
eval_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,
])

current_eval_experiment_id = ""

# --- Helper: Uniform Evaluation (from Script 2) ---
@torch.no_grad()
def evaluate_model_uniformly(model, device_str, num_classes_eval, max_batches_to_eval):
    # ... (Identical to Script 2's evaluate_model_uniformly) ...
    # Minor change to use FIXED_NUM_CLASSES for consistency checks if desired
    global current_eval_experiment_id
    if not os.path.exists(VALIDATION_DATA_PATH):
        print(f"ERROR ({current_eval_experiment_id}): Val data path not found: {VALIDATION_DATA_PATH}")
        return "N/A (Val Data Missing)"
    try:
        val_dataset = ImageFolder(VALIDATION_DATA_PATH, eval_transforms)
        # Consider using FIXED_NUM_CLASSES for this check if all models are meant for it
        if len(val_dataset.classes) != num_classes_eval: # or FIXED_NUM_CLASSES
            print(f"WARNING ({current_eval_experiment_id}): Dataset classes ({len(val_dataset.classes)}) vs Model classes ({num_classes_eval}). Accuracy may be misleading.")
        if len(val_dataset) == 0:
            print(f"WARNING ({current_eval_experiment_id}): Validation dataset at '{VALIDATION_DATA_PATH}' is empty.")
            return 0.0
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False,
                                num_workers=NUM_WORKERS_EVAL, pin_memory=True if device_str=='cuda' else False)
    except Exception as e:
        print(f"ERROR ({current_eval_experiment_id}): Could not load validation data: {e}")
        return f"N/A (Val Data Load Error: {str(e).splitlines()[0]})"

    device_obj_eval = torch.device(device_str) # Renamed to avoid conflict with global DEVICE
    model.to(device_obj_eval); model.eval()
    correct = 0; total = 0; batches_processed = 0
    if max_batches_to_eval == float('inf'):
        print(f"      INFO ({current_eval_experiment_id}): Evaluating on ALL batches on device {device_str}.")
    else:
        print(f"      INFO ({current_eval_experiment_id}): Evaluating on device {device_str} for max {max_batches_to_eval} batches.")

    for images, labels in val_loader:
        images, labels = images.to(device_obj_eval), labels.to(device_obj_eval)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0); correct += (predicted == labels).sum().item()
        batches_processed += 1
        if batches_processed >= max_batches_to_eval: break
    accuracy = (correct / total) if total > 0 else 0.0
    print(f"      INFO ({current_eval_experiment_id}): Accuracy = {accuracy:.4f} ({correct}/{total}) on {batches_processed} batches.")
    return accuracy


# --- Helper Functions: Model File and Size (from Script 2) ---
def get_model_file_path(experiment_path):
    # ... (Identical to Script 2) ...
    pth_files = list(Path(experiment_path).glob("*.pth"))
    if pth_files:
        for common_name in ["model_final.pth", "model_quantized.pth"]:
            for p_file in pth_files:
                if p_file.name == common_name: return p_file
        for p_file in pth_files: # Specific baseline names
            if "baseline_ft_imagenetmini_final.pth" in p_file.name: return p_file
        # Try to find the most recently modified .pth file as a last resort
        # pth_files.sort(key=os.path.getmtime, reverse=True)
        return pth_files[0] # Or the first one found
    return None

def get_model_size_mb(model_path):
    # ... (Identical to Script 2) ...
    if model_path and os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return None

# --- Model Definition and Pruning Application (FROM SCRIPT 1) ---
def get_base_resnet50_model_for_reconstruction(): # Renamed slightly
    model = models.resnet50(weights=None, num_classes=FIXED_NUM_CLASSES)
    return model

def apply_structured_pruning_to_model_for_reconstruction(
    model_to_prune, example_inputs, target_pruning_rate_per_layer, device_obj # Use device_obj
):
    model_to_prune.to(device_obj)
    example_inputs = example_inputs.to(device_obj)
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

def get_pruning_config_from_log_for_reconstruction(log_file_path): # Renamed
    """Helper to load log and extract key pruning param for a single stage/one-shot."""
    if not log_file_path.exists():
        return None
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)

        # For one-shot, directly from config_details
        if 'config_details' in log_data and 'target_filter_pruning_rate_per_layer' in log_data['config_details']:
            rate = log_data['config_details']['target_filter_pruning_rate_per_layer']
            return {'type': 'one-shot', 'rate': float(rate)}

        # For a single iterative stage, get its own applied rate
        if 'config_details' in log_data and 'applied_step_rate_for_this_stage' in log_data['config_details']:
            rate = log_data['config_details']['applied_step_rate_for_this_stage']
            return {'type': 'iterative_step', 'rate': float(rate)}
        
        # Check for overall sparsity if specific step rates are not found (less ideal for reconstruction)
        # This might be needed if iterative logs don't have 'applied_step_rate_for_this_stage' but have a global target
        # However, for *reconstruction*, we need the *exact* per-layer or per-step application.
        # The original script 1 logic seems to rely on 'applied_step_rate_for_this_stage' for iterative.

    except json.JSONDecodeError:
        print(f"    Error decoding JSON from {log_file_path}")
    except Exception as e:
        print(f"    Error processing log {log_file_path}: {e}")
    return None

# This function now needs to be aware of experiment_path to find the log.json
def load_and_reconstruct_structured_pruned_model(model_path_str, experiment_dir_path, device_obj):
    model_path = Path(model_path_str)
    log_file_path = Path(experiment_dir_path) / "log.json" # Assumes log.json is in the exp dir
    
    pruning_rec_config = None
    cumulative_step_rates_for_iterative = [] # For iterative cases

    # Determine if it's iterative and which stage, to build cumulative rates
    # This part is tricky and depends heavily on your naming convention for iterative stages
    # The original Script 1 has a loop for this. Here, we might need to infer based on the current exp_path_str
    # For simplicity, let's assume for now `get_pruning_config_from_log_for_reconstruction` gives enough
    # info, or this function needs to be called in a loop for iterative stages.

    # ---> This is the most complex part to adapt from Script 1's main loop logic <---
    # For a single model path, we need to know its pruning history if it's iterative.
    # Let's assume for now we can get the *final* pruning config directly or it's one-shot.
    # If it's an iterative *final* model, the log should ideally contain the full history or the config
    # needed to reconstruct to *that specific stage*.
    # The provided get_pruning_config_from_log_for_reconstruction seems to get only ONE rate.
    # We need to replicate the iterative accumulation logic from Script 1's main loop if we are processing
    # an intermediate stage of an iterative pruning run.
    #
    # Let's simplify for now: Assume the log.json of the *specific experiment folder* contains
    # the necessary one-shot rate OR if it's iterative, we need to know the history.
    # The provided script 1's `load_reconstructed_pruned_model` handles `pruning_config` with `type: iterative` and `step_rates`.
    # We need to generate that `pruning_config` correctly before calling this.

    # This function should receive the *complete* pruning_rec_config
    # The calling function (process_single_model_file) will need to prepare this.
    # For now, let's keep load_reconstructed_pruned_model similar to Script 1's version
    
    # --- This function is essentially Script 1's load_reconstructed_pruned_model ---
    reconstructed_model = get_base_resnet50_model_for_reconstruction() # Assuming ResNet50 base
    reconstructed_model.to(device_obj)
    # Use INPUT_TENSOR_CPU for example_inputs, ensure it's on the correct device
    example_inputs_local = INPUT_TENSOR_CPU.to(device_obj)

    # The `pruning_config` needs to be passed in or determined here.
    # For now, let's assume the caller will prepare this `pruning_config`.
    # This is a placeholder, this logic needs to be more robust based on log content.
    
    # --- Placeholder for getting the correct pruning_rec_config ---
    # This needs to be made robust by parsing the log in the context of the experiment type (one-shot/iterative)
    # This logic is simplified here and likely needs the iterative handling from script 1's main loop.
    temp_pruning_info_from_log = get_pruning_config_from_log_for_reconstruction(log_file_path)
    pruning_config_for_this_model = None

    if temp_pruning_info_from_log and temp_pruning_info_from_log['type'] == 'one-shot':
        pruning_config_for_this_model = temp_pruning_info_from_log
    elif "iterative" in str(experiment_dir_path).lower() or "it" in str(experiment_dir_path).lower():
        # This is where it gets complex without the full iterative loop context from script 1
        # We'd need to find *all* previous stage logs for this iterative experiment base name
        # and accumulate rates. This is beyond a simple helper here.
        # For now, we'll assume if it's iterative, the log might tell us what to do, or this will fail.
        # A more robust solution would be to pass the accumulated step_rates.
        print(f"    WARNING: Iterative model {model_path_str}. Reconstruction might be incomplete without full stage history.")
        # Attempt to use if log has 'applied_step_rate_for_this_stage', assuming it's the only step needed for this *specific* model file
        if temp_pruning_info_from_log and temp_pruning_info_from_log['type'] == 'iterative_step':
             pruning_config_for_this_model = {'type': 'iterative', 'step_rates': [temp_pruning_info_from_log['rate']]} # Simplification!
        else:
            print(f"    Could not determine iterative pruning steps for {model_path_str} from its log alone.")
            return None # Cannot reconstruct
    # --- End Placeholder ---

    if not pruning_config_for_this_model:
        print(f"    ERROR: Could not determine pruning config for reconstruction for {model_path_str} from {log_file_path}")
        return None

    try:
        if pruning_config_for_this_model['type'] == 'one-shot':
            rate = pruning_config_for_this_model['rate']
            reconstructed_model = apply_structured_pruning_to_model_for_reconstruction(
                reconstructed_model, example_inputs_local, rate, device_obj)
        elif pruning_config_for_this_model['type'] == 'iterative':
            step_rates = pruning_config_for_this_model['step_rates']
            current_arch_model = reconstructed_model
            for i, step_rate in enumerate(step_rates):
                # print(f"      Applying iterative step {i+1} with rate {step_rate} for reconstruction")
                current_arch_model = apply_structured_pruning_to_model_for_reconstruction(
                    current_arch_model, example_inputs_local, step_rate, device_obj)
            reconstructed_model = current_arch_model
        else:
            print(f"    ERROR: Unknown pruning_config type: {pruning_config_for_this_model['type']}")
            return None

        state_dict = torch.load(model_path_str, map_location=device_obj)
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Handle cases where state_dict might be nested (common in some saving practices)
        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
            state_dict = state_dict['state_dict']

        reconstructed_model.load_state_dict(state_dict)
        print(f"    State_dict loaded successfully into RECONSTRUCTED structured model: {model_path_str}")
        reconstructed_model.eval()
        return reconstructed_model
    except Exception as e:
        print(f"    ERROR loading/reconstructing structured pruned model {model_path_str}: {e}")
        import traceback; traceback.print_exc()
        return None


# --- Core Function: Process Single Model (MODIFIED) ---
def process_single_model_file(
    model_path, # Path to the .pth file
    experiment_path_str, # Path to the experiment's directory (to find log.json)
    base_arch,
    num_classes_model,
    experiment_id_str,
    config_details_dict,
    is_structured_pruning_exp # New flag
    ):
    global current_eval_experiment_id, baseline_metrics
    current_eval_experiment_id = experiment_id_str

    results = {
        'Final_Val_Accuracy_Uniform': "N/A (Processing Error)",
        'Inference_Time_ms_CPU (Batch 1)': "N/A",
        'Inference_Time_ms_GPU (Batch 1)': "N/A (CUDA unavailable or error)",
        'FLOPs_GMACs': "N/A",
        'Params_Millions': "N/A"
    }

    if not model_path or not os.path.exists(model_path):
        for key in results: results[key] = "N/A (No model file)"; return results
    if os.path.getsize(model_path) == 0:
        for key in results: results[key] = "N/A (Model file 0 bytes)"; return results

    loaded_model_obj = None
    model_for_gpu_timing = None # Separate model for GPU timing if reconstructed
    model_load_error_msg_specific = None
    load_method_used = "Unknown"

    # --- Standard Loading (for FLOPs, Accuracy, CPU time, and non-structured GPU time) ---
    try:
        _loaded_content = None
        try:
            # Try JIT load first, as it might be a JIT-saved pruned model
            loaded_model_obj = torch.jit.load(model_path, map_location='cpu')
            load_method_used = "torch.jit.load()"
            # print(f"      DEBUG ({experiment_id_str}): Model (potentially JIT) loaded using {load_method_used}.")
        except Exception: pass # Silently try next

        if loaded_model_obj is None:
            try:
                _loaded_full_model = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(_loaded_full_model, torch.nn.Module):
                    loaded_model_obj = _loaded_full_model
                    load_method_used = "torch.load() [full model]"
                    # print(f"      DEBUG ({experiment_id_str}): Model (full nn.Module) loaded using {load_method_used}.")
                elif isinstance(_loaded_full_model, dict):
                    _loaded_content = _loaded_full_model
                    load_method_used = "torch.load() [content is dict]"
                else:
                    model_load_error_msg_specific = f"N/A (Loaded obj not nn.Module/dict: {type(_loaded_full_model)})"
                    raise ValueError(model_load_error_msg_specific)
            except Exception as e_full_load:
                 if loaded_model_obj is None and _loaded_content is None:
                    try:
                        _loaded_content = torch.load(model_path, map_location='cpu', weights_only=False)
                        if not isinstance(_loaded_content, dict):
                             raise e_full_load
                        load_method_used = "torch.load() [content is dict, after initial fail]"
                    except Exception as e_final_load:
                         model_load_error_msg_specific = f"N/A (File Load Fail: {e_final_load})"
                         raise ValueError(model_load_error_msg_specific)

        if loaded_model_obj is None: # Must be a state_dict
            if not isinstance(_loaded_content, dict):
                 model_load_error_msg_specific = "N/A (File not JIT/Full/StateDict after attempts)"
                 raise ValueError(model_load_error_msg_specific)

            # For non-structured pruning, or if structured reconstruction fails, try standard loading
            if not is_structured_pruning_exp: # Only do this if NOT structured or if reconstruction is separate
                print(f"      INFO ({experiment_id_str}): Attempting standard state_dict load for non-structured or as fallback.")
                if base_arch == "ResNet18": model_instance = models.resnet18(weights=None, num_classes=num_classes_model)
                elif base_arch == "ResNet50": model_instance = models.resnet50(weights=None, num_classes=num_classes_model)
                else:
                    model_load_error_msg_specific = f"N/A (Unknown base_arch '{base_arch}')"
                    raise ValueError(model_load_error_msg_specific)

                state_dict_to_load = _loaded_content
                if any(k.startswith('module.') for k in state_dict_to_load.keys()):
                    state_dict_to_load = {k.replace('module.', ''): v for k, v in state_dict_to_load.items()}
                if 'model' in state_dict_to_load and isinstance(state_dict_to_load['model'], dict):
                    state_dict_to_load = state_dict_to_load['model']
                elif 'state_dict' in state_dict_to_load and isinstance(state_dict_to_load['state_dict'], dict):
                    state_dict_to_load = state_dict_to_load['state_dict']
                
                try:
                    model_instance.load_state_dict(state_dict_to_load)
                    loaded_model_obj = model_instance
                    load_method_used = "torch.load() [state_dict into fresh base model]"
                    # print(f"      DEBUG ({experiment_id_str}): Model (state_dict) loaded using {load_method_used}.")
                except RuntimeError as e_state_dict:
                    if "Error(s) in loading state_dict" in str(e_state_dict) and not is_structured_pruning_exp:
                        # This is an unexpected error for non-structured pruning
                        model_load_error_msg_specific = f"N/A (StateDict Mismatch for non-structured: {e_state_dict})"
                        raise ValueError(model_load_error_msg_specific)
                    elif "Error(s) in loading state_dict" in str(e_state_dict) and is_structured_pruning_exp:
                        # This is expected if we haven't reconstructed yet, will be handled by reconstruction path
                        print(f"      INFO ({experiment_id_str}): State_dict load failed for structured model (expected before reconstruction). Will attempt reconstruction.")
                        pass # Let the structured pruning path handle it
                    else:
                        raise e_state_dict


        if loaded_model_obj is None and not is_structured_pruning_exp: # If still not loaded and not structured
             model_load_error_msg_specific = "N/A (Model not loaded after all non-structured attempts)"
             raise ValueError(model_load_error_msg_specific)
        
        if loaded_model_obj:
            loaded_model_obj.eval()
            # print(f"      DEBUG ({experiment_id_str}): Primary model object prepared (pre-reconstruction if structured).")

    except Exception as e:
        error_detail = str(e).splitlines()[0]
        final_error_msg = model_load_error_msg_specific if model_load_error_msg_specific else f"N/A (Model Load Error: {error_detail})"
        if "Error(s) in loading state_dict for ResNet" in str(e) and not is_structured_pruning_exp:
            final_error_msg = "N/A (Arch mismatch/StateDict load fail for non-structured)"
        elif "Unsupported qscheme" in str(e) or "SerializedAttributes" in str(e):
            final_error_msg = f"N/A (Quantized Load Err: {error_detail})"
        
        # If it's a structured pruning experiment and primary loading failed, it's okay, reconstruction is next
        if is_structured_pruning_exp and "Error(s) in loading state_dict" in str(e):
            print(f"      INFO ({experiment_id_str}): Primary load failed for structured model, proceeding to reconstruction attempt.")
        elif is_structured_pruning_exp and loaded_model_obj is None : # if some other error before reconstruction
             print(f"      WARNING ({experiment_id_str}): Some loading error before structured reconstruction: {final_error_msg}")
        else: # Non-structured model load failure is terminal for this model
            for key in results: results[key] = final_error_msg
            if torch.cuda.is_available() and final_error_msg.startswith("N/A ("): results['Inference_Time_ms_GPU (Batch 1)'] = final_error_msg
            return results
    
    # --- Model for FLOPs, Params, Accuracy, CPU Time ---
    # For structured pruning, if loaded_model_obj is JIT, it *might* already have the pruned structure.
    # If loaded_model_obj is None (state_dict load failed for structured), we *must* reconstruct.
    # If loaded_model_obj is an nn.Module (e.g. from torch.save(full_model)), it should be pruned.
    
    model_for_measurement = None
    if is_structured_pruning_exp:
        print(f"      INFO ({experiment_id_str}): Structured pruning experiment. Attempting reconstruction for consistent measurements.")
        # For structured pruning, ALWAYS try to reconstruct for FLOPs/Params/Accuracy/CPU to be like Script 1.
        # The experiment_path_str is the directory of the experiment.
        reconstructed_for_measure = load_and_reconstruct_structured_pruned_model(
            model_path, experiment_path_str, torch.device('cpu') # Reconstruct on CPU for thop
        )
        if reconstructed_for_measure:
            model_for_measurement = reconstructed_for_measure
            print(f"      INFO ({experiment_id_str}): Successfully reconstructed structured model for CPU measurements.")
            # This reconstructed model will also be used for GPU timing later if DEVICE is cuda
            if DEVICE.type == 'cuda':
                model_for_gpu_timing = reconstructed_for_measure # Use the same reconstructed for GPU
        else:
            print(f"      WARNING ({experiment_id_str}): Failed to reconstruct structured model. Falling back to loaded_model_obj if available.")
            if loaded_model_obj:
                print(f"        Using the initially loaded object (type: {type(loaded_model_obj)}) for measurements. FLOPs/Params might be from JIT.")
                model_for_measurement = loaded_model_obj
            else:
                results['FLOPs_GMACs'] = "N/A (Struct Reconstruct Fail)"
                results['Params_Millions'] = "N/A (Struct Reconstruct Fail)"
                results['Final_Val_Accuracy_Uniform'] = "N/A (Struct Reconstruct Fail)"
                # GPU time will also be N/A
    else: # Not structured pruning
        model_for_measurement = loaded_model_obj

    if model_for_measurement is None: # If all loading/reconstruction failed
        print(f"      ERROR ({experiment_id_str}): No valid model object available after all attempts.")
        for key in results: results[key] = "N/A (Model Unavailable)"; return results


    # --- FLOPs and Params Calculation ---
    flops_calculated_by_thop = False
    if isinstance(model_for_measurement, torch.nn.Module) and not isinstance(model_for_measurement, torch.jit.ScriptModule):
        try:
            # model_for_flops_params = model_for_measurement.to(torch.device('cpu')) # Already on CPU if reconstructed
            dummy_input_flops = INPUT_TENSOR_CPU.to(torch.device('cpu'))
            macs, params = profile(model_for_measurement, inputs=(dummy_input_flops,), verbose=False)
            results['FLOPs_GMACs'] = macs / 1e9
            results['Params_Millions'] = params / 1e6
            flops_calculated_by_thop = True
            # del model_for_flops_params # No, keep model_for_measurement
        except Exception as e_flops:
            print(f"      WARNING ({experiment_id_str}): Could not calculate FLOPs/Params with thop: {e_flops}")
            pass
    elif isinstance(model_for_measurement, torch.jit.ScriptModule):
         print(f"      INFO ({experiment_id_str}): FLOPs/Params: model is JIT ScriptModule. Thop skipped. Baseline fallback if applicable.")

    # FLOPs/Params Fallback (Identical to Script 2)
    is_ao_quant_or_kmeans = "ptq" in experiment_id_str.lower() or \
                            "qat" in experiment_id_str.lower() or \
                            "kmeans" in experiment_id_str.lower()
    if not flops_calculated_by_thop and (is_ao_quant_or_kmeans or isinstance(model_for_measurement, torch.jit.ScriptModule)):
        if base_arch in baseline_metrics and baseline_metrics[base_arch]:
            baseline_f = baseline_metrics[base_arch].get("flops_gmacs")
            baseline_p = baseline_metrics[base_arch].get("params_millions")
            if pd.notna(baseline_f): results['FLOPs_GMACs'] = baseline_f
            else: results['FLOPs_GMACs'] = "N/A (Baseline FLOPs Missing)"
            if pd.notna(baseline_p): results['Params_Millions'] = baseline_p
            else: results['Params_Millions'] = "N/A (Baseline Params Missing)"
        else:
            results['FLOPs_GMACs'] = "N/A (Thop Skip/Baseline Miss)"
            results['Params_Millions'] = "N/A (Thop Skip/Baseline Miss)"
    elif not flops_calculated_by_thop and results['FLOPs_GMACs'] == "N/A": # Only if not set by fallback
        results['FLOPs_GMACs'] = "N/A (FLOPs Error)"
        results['Params_Millions'] = "N/A (Params Error)"


    # --- Uniform Accuracy Evaluation ---
    is_gpu_unstable_model = experiment_id_str in GPU_UNSTABLE_QUANTIZED_MODELS
    eval_device_for_acc = 'cpu' if is_gpu_unstable_model else DEVICE.type # Use global DEVICE
    if is_gpu_unstable_model:
        print(f"      INFO ({experiment_id_str}): Known GPU unstable. Forcing CPU evaluation for accuracy.")

    # Ensure model_for_measurement is on the correct device for accuracy eval
    model_for_accuracy_eval = model_for_measurement.to(torch.device(eval_device_for_acc))
    accuracy_val = evaluate_model_uniformly(model_for_accuracy_eval, eval_device_for_acc, num_classes_model, MAX_EVAL_BATCHES)
    results['Final_Val_Accuracy_Uniform'] = accuracy_val
    if model_for_accuracy_eval is not model_for_measurement : del model_for_accuracy_eval # if a copy was made


    # --- CPU Inference Timing ---
    cpu_model_for_timing = model_for_measurement.to(torch.device('cpu'))
    try:
        with torch.no_grad():
            for _ in range(WARMUP_INFERENCES): _ = cpu_model_for_timing(INPUT_TENSOR_CPU)
            timings_cpu = []
            for _ in range(TIMED_INFERENCES):
                start_time = time.perf_counter(); _ = cpu_model_for_timing(INPUT_TENSOR_CPU); end_time = time.perf_counter()
                timings_cpu.append((end_time - start_time) * 1000)
        results['Inference_Time_ms_CPU (Batch 1)'] = sum(timings_cpu) / len(timings_cpu) if timings_cpu else "N/A (CPU Time Error)"
    except Exception as e_cpu_time: results['Inference_Time_ms_CPU (Batch 1)'] = f"N/A (CPU Time Error: {str(e_cpu_time).splitlines()[0]})"
    # Don't delete cpu_model_for_timing if it's the same as model_for_measurement

    # --- GPU Inference Timing (Corrected Logic) ---
    if is_gpu_unstable_model:
        results['Inference_Time_ms_GPU (Batch 1)'] = "N/A (Known JIT GPU Unstable)"
    elif DEVICE.type == 'cuda' and INPUT_TENSOR_GPU is not None:
        # Determine which model to use for GPU timing
        target_model_for_gpu = None
        if is_structured_pruning_exp and model_for_gpu_timing: # model_for_gpu_timing was set during reconstruction
            print(f"      INFO ({experiment_id_str}): Using RECONSTRUCTED structured model for GPU timing.")
            target_model_for_gpu = model_for_gpu_timing.to(DEVICE)
        elif model_for_measurement: # Use the general measurement model (could be JIT or nn.Module)
            print(f"      INFO ({experiment_id_str}): Using initially loaded/standard model for GPU timing.")
            target_model_for_gpu = model_for_measurement.to(DEVICE)
        
        if target_model_for_gpu:
            try:
                with torch.no_grad():
                    for _ in range(WARMUP_INFERENCES): _ = target_model_for_gpu(INPUT_TENSOR_GPU); torch.cuda.synchronize()
                    timings_gpu = []
                    for _ in range(TIMED_INFERENCES):
                        torch.cuda.synchronize(); start_time = time.perf_counter(); _ = target_model_for_gpu(INPUT_TENSOR_GPU); torch.cuda.synchronize(); end_time = time.perf_counter()
                        timings_gpu.append((end_time - start_time) * 1000)
                results['Inference_Time_ms_GPU (Batch 1)'] = sum(timings_gpu) / len(timings_gpu) if timings_gpu else "N/A (GPU Time Error)"
            except Exception as e_gpu_time: results['Inference_Time_ms_GPU (Batch 1)'] = f"N/A (GPU Time Error: {str(e_gpu_time).splitlines()[0]})"
            if target_model_for_gpu is not model_for_measurement and target_model_for_gpu is not model_for_gpu_timing:
                 del target_model_for_gpu # Delete if it was a separate copy for GPU
        else:
            results['Inference_Time_ms_GPU (Batch 1)'] = "N/A (No model for GPU timing)"
            
    elif DEVICE.type != 'cuda': results['Inference_Time_ms_GPU (Batch 1)'] = "N/A (CUDA unavailable)"

    # --- Cleanup ---
    if model_for_measurement: del model_for_measurement
    if model_for_gpu_timing: del model_for_gpu_timing # This might be same as reconstructed_for_measure
    if 'loaded_model_obj' in locals() and loaded_model_obj: del loaded_model_obj
    if 'cpu_model_for_timing' in locals() and cpu_model_for_timing: del cpu_model_for_timing
    # Add other specific dels if necessary

    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    return results


# --- Main Processing Logic (MODIFIED to pass experiment_path_str and is_structured_pruning_exp) ---
all_experiments_data = []
baseline_metrics = {}
# ... (Validation path check from Script 2) ...

print("\n--- Processing baselines ---")
# ... (Baseline processing loop from Script 2, mostly unchanged) ...
# In process_single_model_file call for baselines, set is_structured_pruning_exp=False
for cat_name_outer in os.listdir(ROOT_DIR):
    cat_path_outer = os.path.join(ROOT_DIR, cat_name_outer)
    if os.path.isdir(cat_path_outer) and ("baseline" in cat_name_outer.lower()):
        exp_name = cat_name_outer
        print(f"Processing baseline experiment: {exp_name}")
        exp_path = cat_path_outer
        row = {"Experiment_ID": exp_name}
        # ... (rest of baseline row population) ...
        if "resnet18" in exp_name.lower(): row["Base_Model_Arch"] = "ResNet18"
        elif "resnet50" in exp_name.lower(): row["Base_Model_Arch"] = "ResNet50"
        else: row["Base_Model_Arch"] = "Unknown"
        row["Optimization_Category"] = "Baseline"; row["Specific_Technique"] = "Baseline"; row["Key_Parameters"] = "N/A"
        log_path = os.path.join(exp_path, "log.json"); model_file_path = get_model_file_path(exp_path)
        log_data, config_details, training_summary, original_eval_metrics = {}, {}, {}, {}
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f: log_data = json.load(f)
            except json.JSONDecodeError: print(f"Error decoding JSON for {log_path}")
        config_details = log_data.get('config_details', {}); training_summary = log_data.get('training_summary', {})
        original_eval_metrics = log_data.get('original_evaluation_metrics_from_log', {})
        row["Accuracy_Drop_From_Best_Epoch_pp"] = "N/A"; row["Accuracy_Before_FT"] = training_summary.get('accuracy_before_ft', "N/A")
        row["Model_Size_MB_Log"] = original_eval_metrics.get('model_size_mb', "N/A"); row["Achieved_Sparsity_Percent"] = "N/A"
        row["FT_Epochs_Run"] = training_summary.get('num_epochs_trained', 0)
        ft_epochs_run_val = row["FT_Epochs_Run"] if isinstance(row["FT_Epochs_Run"], (int, float)) and pd.notna(row["FT_Epochs_Run"]) else 0
        row["FT_Time_seconds"] = training_summary.get('total_training_time_seconds', 0.0) if ft_epochs_run_val > 0 else 0.0
        row["Notes_from_Log"] = training_summary.get('notes', ''); row["Model_Size_MB_Disk"] = get_model_size_mb(model_file_path)
        num_classes_model = config_details.get('num_classes', DEFAULT_NUM_CLASSES)

        measured_metrics = process_single_model_file(
            model_file_path,
            exp_path, # experiment_path_str
            row["Base_Model_Arch"],
            num_classes_model,
            exp_name,
            config_details,
            is_structured_pruning_exp=False # Baselines are not structured-pruned
        )
        row.update(measured_metrics)
        if row["Base_Model_Arch"] != "Unknown":
            baseline_metrics[row["Base_Model_Arch"]] = {
                "val_accuracy": pd.to_numeric(row["Final_Val_Accuracy_Uniform"], errors='coerce'),
                "model_size_mb_disk": pd.to_numeric(row["Model_Size_MB_Disk"], errors='coerce'),
                "inference_cpu_ms": pd.to_numeric(row["Inference_Time_ms_CPU (Batch 1)"], errors='coerce'),
                "inference_gpu_ms": pd.to_numeric(row["Inference_Time_ms_GPU (Batch 1)"], errors='coerce'),
                "flops_gmacs": pd.to_numeric(row["FLOPs_GMACs"], errors='coerce'),
                "params_millions": pd.to_numeric(row["Params_Millions"], errors='coerce')
            }
        all_experiments_data.append(row)


print("\n--- Processing other experiments ---")
for cat_name in os.listdir(ROOT_DIR):
    cat_path = os.path.join(ROOT_DIR, cat_name)
    if not os.path.isdir(cat_path) or "baseline" in cat_name.lower(): continue
    print(f"Processing category: {cat_name}")

    # Check if this category is for structured pruning
    is_cat_structured_pruning = "pruning_structured_iterative" in cat_name.lower() or \
                                "pruning_structured_oneshot" in cat_name.lower()

    for exp_name in os.listdir(cat_path):
        exp_path = os.path.join(cat_path, exp_name) # This is experiment_path_str
        if not os.path.isdir(exp_path): continue
        print(f"  Processing experiment: {exp_name}"); row = {"Experiment_ID": exp_name}
        # ... (rest of row population from Script 2, like Base_Model_Arch, Optimization_Category, etc.) ...
        # Determine base_arch, num_classes, config_details as in Script 2
        if "resnet18" in exp_name.lower(): row["Base_Model_Arch"] = "ResNet18"
        elif "resnet50" in exp_name.lower(): row["Base_Model_Arch"] = "ResNet50"
        else:
            if cat_name == "combined_distilled_quantized": row["Base_Model_Arch"] = "ResNet18"
            elif "pruning_structured" in cat_name.lower() and "resnet50" in exp_name.lower() : row["Base_Model_Arch"] = "ResNet50" # Be more specific for structured
            else: row["Base_Model_Arch"] = "ResNet50" # Default or adjust as needed

        opt_cat_map = {"combined_distilled_quantized": "Combined", "knowledge_distillation": "Knowledge Distillation","pruning_nm_sparsity": "Pruning", "pruning_structured_iterative": "Pruning","pruning_structured_oneshot": "Pruning", "pruning_unstructured_iterative": "Pruning","pruning_unstructured_oneshot": "Pruning", "quantization_kmeans": "Quantization","quantization_ptq_int8": "Quantization", "quantization_qat_int8": "Quantization",}
        row["Optimization_Category"] = opt_cat_map.get(cat_name, "Other")
        log_path = os.path.join(exp_path, "log.json"); model_file_path = get_model_file_path(exp_path)
        log_data, config_details, training_summary, original_eval_metrics, quant_specific_details = {}, {}, {}, {}, {}
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f: log_data = json.load(f)
            except json.JSONDecodeError:
                print(f"    Error decoding JSON for {log_path}")
                # Fill essential error placeholders
                for key_to_fill in ["Specific_Technique", "Key_Parameters", "Final_Val_Accuracy_Uniform", "Model_Size_MB_Disk", "Inference_Time_ms_CPU (Batch 1)", "FLOPs_GMACs", "Params_Millions"]: row[key_to_fill] = "JSON Error"
                all_experiments_data.append(row); continue
        config_details = log_data.get('config_details', {}); training_summary = log_data.get('training_summary', {}); original_eval_metrics = log_data.get('original_evaluation_metrics_from_log', {}); quant_specific_details = log_data.get('quantization_specific_details', {})
        
        # Populate Specific_Technique, Key_Parameters etc. as in Script 2
        specific_tech_parts, key_params_parts = [], []
        # ... (copy this logic from your script 2 accurately) ...
        if config_details.get('teacher_model_architecture'):
            specific_tech_parts.append("Knowledge Distillation"); teacher = config_details.get('teacher_model_architecture'); student = config_details.get('student_model_architecture', row["Base_Model_Arch"])
            key_params_parts.append(f"T:{teacher}->S:{student}");
            if row["Base_Model_Arch"] == "ResNet50" and "resnet18" in student.lower(): row["Base_Model_Arch"] = "ResNet18"
        quant_method_cfg = str(config_details.get('quantization_method_type', '')).lower()
        if "kmeans" in quant_method_cfg or "kmeans" in exp_name.lower():
            specific_tech_parts.append("KMeans Quant"); clusters = config_details.get('kmeans_clusters') or quant_specific_details.get('kmeans_clusters')
            if clusters: key_params_parts.append(f"Clusters: {clusters}")
        elif "ptq" in quant_method_cfg or ("quant" in exp_name.lower() and "ptq" in exp_name.lower()):
            tech = "PTQ INT8"
            if "per_channel" in quant_method_cfg or "perchannel" in exp_name.lower(): tech += " (Per-Channel)"
            elif "per_tensor" in quant_method_cfg or "pertensor" in exp_name.lower(): tech += " (Per-Tensor)"
            else:
                 if "perchannel" in exp_name.lower(): tech += " (Per-Channel)"
                 elif "pertensor" in exp_name.lower(): tech += " (Per-Tensor)"
            specific_tech_parts.append(tech)
        elif "qat" in quant_method_cfg or ("quant" in exp_name.lower() and "qat" in exp_name.lower()):
            specific_tech_parts.append("QAT INT8"); epochs = config_details.get('qat_epochs')
            if epochs is not None: key_params_parts.append(f"QAT Epochs: {epochs}")
        pruning_tech_exp_name = exp_name.lower(); pruning_method_cfg = config_details.get('pruning_method_name', '').lower(); pruning_strat_cfg = config_details.get('pruning_strategy_type', '').lower()
        
        current_exp_is_structured = False
        if "prune_struct_it" in pruning_tech_exp_name or "iterative_structured" in pruning_strat_cfg:
            specific_tech_parts.append("Iterative Structured Pruning (L1 Filter)")
            current_exp_is_structured = True
            # For iterative structured, the key_params (rates) might be complex.
            # Script 1's main loop builds this cumulatively. Here we might get only the current stage.
            # The reconstruction logic will try its best with the log of the *current* stage.
            # This is a known simplification for this hybrid script unless full iterative parsing is added.
            log_pr_conf = get_pruning_config_from_log_for_reconstruction(Path(log_path))
            if log_pr_conf and log_pr_conf.get('type') == 'iterative_step':
                 key_params_parts.append(f"Stage Rate: {log_pr_conf['rate']*100:.1f}% (Note: full history for reconstruction is complex)")
            elif log_pr_conf and log_pr_conf.get('type') == 'one-shot': # Should not happen for iterative name
                 key_params_parts.append(f"Rate: {log_pr_conf['rate']*100:.1f}%")


        elif "prune_struct_os" in pruning_tech_exp_name or "one_shot_structured" in pruning_strat_cfg or "structured_l1_filter" in pruning_method_cfg :
            specific_tech_parts.append("One-Shot Structured Pruning (L1 Filter)")
            current_exp_is_structured = True
            log_pr_conf = get_pruning_config_from_log_for_reconstruction(Path(log_path))
            if log_pr_conf and log_pr_conf.get('type') == 'one-shot':
                 key_params_parts.append(f"Rate: {log_pr_conf['rate']*100:.1f}%")

        elif "prune_nm" in pruning_tech_exp_name or "nm_sparsity" in pruning_method_cfg:
            if "N:M Sparsity" not in specific_tech_parts: specific_tech_parts.append("N:M Sparsity")
            n_val = config_details.get('nm_sparsity_n', 2); m_val = config_details.get('nm_sparsity_m', 4); key_params_parts.append(f"N:{n_val}, M:{m_val}")
        elif "prune_unstruct_it" in pruning_tech_exp_name or "iterative_unstructured" in pruning_strat_cfg : specific_tech_parts.append("Iterative Unstructured Pruning (L1)")
        elif "prune_unstruct_os" in pruning_tech_exp_name or "one_shot_unstructured" in pruning_strat_cfg : specific_tech_parts.append("One-Shot Unstructured Pruning (L1)")

        if any("Pruning" in tech for tech in specific_tech_parts) and not current_exp_is_structured: # Only if not already handled by structured specific
            target_sparsities = [config_details.get('target_overall_sparsity_approx_for_this_stage'), config_details.get('target_filter_pruning_rate_per_layer'), config_details.get('target_sparsity_for_this_stage'), config_details.get('target_sparsity')]
            for sp_val in target_sparsities:
                if sp_val is not None:
                    try: key_params_parts.append(f"Target Sparsity: {float(sp_val)*100:.1f}%")
                    except ValueError: key_params_parts.append(f"Target Sparsity: {sp_val}")
                    break
        row["Specific_Technique"] = " + ".join(list(dict.fromkeys(specific_tech_parts))) if specific_tech_parts else "Other"; row["Key_Parameters"] = "; ".join(key_params_parts) if key_params_parts else "N/A"

        # ... (populate other log-based fields like Accuracy_Before_FT, Model_Size_MB_Log, etc.) ...
        row["Accuracy_Drop_From_Best_Epoch_pp"] = "N/A" # Or from log if available
        acc_before_ft = training_summary.get('accuracy_before_ft');
        if acc_before_ft is None: acc_before_ft = training_summary.get('accuracy_before_ft_this_stage')
        if acc_before_ft is None: acc_before_ft = training_summary.get('evaluation_accuracy_after_pruning_before_ft')
        row["Accuracy_Before_FT"] = acc_before_ft if acc_before_ft is not None else "N/A"; row["Model_Size_MB_Log"] = original_eval_metrics.get('model_size_mb', "N/A")
        ach_sparsity = training_summary.get('achieved_overall_parameter_sparsity_percent');
        if ach_sparsity is None: ach_sparsity = training_summary.get('achieved_overall_sparsity_percent_after_stage')
        row["Achieved_Sparsity_Percent"] = ach_sparsity if ach_sparsity is not None else "N/A"
        num_epochs_trained_val = training_summary.get('num_epochs_trained') or training_summary.get('num_epochs_trained_in_stage',0)
        row["FT_Epochs_Run"] = num_epochs_trained_val if num_epochs_trained_val is not None else 0
        ft_time = training_summary.get('total_training_time_seconds');
        if ft_time is None: ft_time = training_summary.get('total_training_time_seconds_in_stage')
        row["FT_Time_seconds"] = ft_time if ft_time is not None and row["FT_Epochs_Run"] > 0 else 0.0
        row["Notes_from_Log"] = training_summary.get('notes', ''); row["Model_Size_MB_Disk"] = get_model_size_mb(model_file_path)

        num_classes_model = config_details.get('num_classes', DEFAULT_NUM_CLASSES)
        if 'student_config' in config_details and isinstance(config_details['student_config'], dict):
            num_classes_model = config_details['student_config'].get('num_classes', num_classes_model)


        # Pass the experiment directory path (exp_path)
        measured_metrics = process_single_model_file(
            model_file_path,
            exp_path, # experiment_path_str (directory)
            row["Base_Model_Arch"],
            num_classes_model,
            exp_name,
            config_details,
            is_structured_pruning_exp=current_exp_is_structured # Pass the flag
        )
        row.update(measured_metrics)
        all_experiments_data.append(row)

# --- Create DataFrame & Finalize Columns (Identical to Script 2) ---
df = pd.DataFrame(all_experiments_data)
if "Final_Val_Accuracy_Uniform" in df.columns:
    df.rename(columns={"Final_Val_Accuracy_Uniform": "Final_Val_Accuracy"}, inplace=True)

desired_columns = [
    "Experiment_ID", "Base_Model_Arch", "Optimization_Category", "Specific_Technique", "Key_Parameters",
    "Final_Val_Accuracy", "Accuracy_Drop_From_Best_Epoch_pp", "Accuracy_Before_FT",
    "Model_Size_MB_Disk", "Model_Size_MB_Log",
    "Params_Millions", "FLOPs_GMACs",
    "Achieved_Sparsity_Percent", "FT_Epochs_Run", "FT_Time_seconds",
    "Inference_Time_ms_CPU (Batch 1)", "Inference_Time_ms_GPU (Batch 1)", "Notes_from_Log",
    "Baseline_Val_Accuracy", "Accuracy_Change_vs_Baseline_pp", "Accuracy_Retention_Percent",
    "Baseline_Model_Size_MB_Disk", "Model_Size_Reduction_vs_Baseline_Percent",
    "Baseline_Params_Millions", "Params_Reduction_vs_Baseline_Percent",
    "Baseline_FLOPs_GMACs", "FLOPs_Reduction_vs_Baseline_Percent",
    "Baseline_Inference_Time_ms_CPU", "Inference_Speedup_vs_Baseline_CPU"
]
if torch.cuda.is_available():
    desired_columns.extend(["Baseline_Inference_Time_ms_GPU", "Inference_Speedup_vs_Baseline_GPU"])

# Ensure all desired columns exist, fill with pd.NA if not
for col in desired_columns:
    if col not in df.columns:
        df[col] = pd.NA # Use pandas' NA for missing data

df = df.reindex(columns=desired_columns) # Reorder and add any missing

# --- Calculate Relative Metrics (Identical to Script 2) ---
print("\n--- Calculating relative metrics ---")
# ... (Copy the relative metrics calculation loop from Script 2) ...
for index, row_series in df.iterrows():
    # Determine baseline arch (as in script 2)
    baseline_arch_to_use = "ResNet50" # Default
    opt_cat_str = str(row_series.get("Optimization_Category","")).strip()
    exp_id_str_lower = str(row_series.get("Experiment_ID","")).lower()
    base_model_arch_str = str(row_series.get("Base_Model_Arch", "")).strip()

    if opt_cat_str in ["Knowledge Distillation", "Combined"]:
        baseline_arch_to_use = "ResNet18"
    elif "resnet18" in exp_id_str_lower and base_model_arch_str == "ResNet18":
        baseline_arch_to_use = "ResNet18"
    # else: use default ResNet50

    if opt_cat_str == "Baseline":
        df.loc[index, "Baseline_Val_Accuracy"] = pd.to_numeric(row_series.get("Final_Val_Accuracy"), errors='coerce')
        df.loc[index, "Accuracy_Change_vs_Baseline_pp"] = 0.0; df.loc[index, "Accuracy_Retention_Percent"] = 100.0
        df.loc[index, "Baseline_Model_Size_MB_Disk"] = pd.to_numeric(row_series.get("Model_Size_MB_Disk"), errors='coerce')
        df.loc[index, "Model_Size_Reduction_vs_Baseline_Percent"] = 0.0
        df.loc[index, "Baseline_Params_Millions"] = pd.to_numeric(row_series.get("Params_Millions"), errors='coerce')
        df.loc[index, "Params_Reduction_vs_Baseline_Percent"] = 0.0
        df.loc[index, "Baseline_FLOPs_GMACs"] = pd.to_numeric(row_series.get("FLOPs_GMACs"), errors='coerce')
        df.loc[index, "FLOPs_Reduction_vs_Baseline_Percent"] = 0.0
        df.loc[index, "Baseline_Inference_Time_ms_CPU"] = pd.to_numeric(row_series.get("Inference_Time_ms_CPU (Batch 1)"), errors='coerce')
        df.loc[index, "Inference_Speedup_vs_Baseline_CPU"] = 1.0
        if torch.cuda.is_available() and "Baseline_Inference_Time_ms_GPU" in df.columns:
            df.loc[index, "Baseline_Inference_Time_ms_GPU"] = pd.to_numeric(row_series.get("Inference_Time_ms_GPU (Batch 1)"), errors='coerce')
            df.loc[index, "Inference_Speedup_vs_Baseline_GPU"] = 1.0
        continue

    if baseline_arch_to_use not in baseline_metrics or not baseline_metrics[baseline_arch_to_use]:
        print(f"Warning: Baseline metrics for {baseline_arch_to_use} not found for experiment {row_series.get('Experiment_ID')}. Skipping relative metrics.")
        continue

    current_baseline = baseline_metrics[baseline_arch_to_use]
    baseline_acc = pd.to_numeric(current_baseline.get("val_accuracy"), errors='coerce')
    baseline_size_disk = pd.to_numeric(current_baseline.get("model_size_mb_disk"), errors='coerce')
    baseline_params = pd.to_numeric(current_baseline.get("params_millions"), errors='coerce')
    baseline_flops = pd.to_numeric(current_baseline.get("flops_gmacs"), errors='coerce')
    baseline_infer_cpu = pd.to_numeric(current_baseline.get("inference_cpu_ms"), errors='coerce')

    df.loc[index, "Baseline_Val_Accuracy"] = baseline_acc
    df.loc[index, "Baseline_Model_Size_MB_Disk"] = baseline_size_disk
    df.loc[index, "Baseline_Params_Millions"] = baseline_params
    df.loc[index, "Baseline_FLOPs_GMACs"] = baseline_flops
    df.loc[index, "Baseline_Inference_Time_ms_CPU"] = baseline_infer_cpu

    final_acc = pd.to_numeric(row_series.get("Final_Val_Accuracy"), errors='coerce')
    if pd.notna(final_acc) and pd.notna(baseline_acc):
        df.loc[index, "Accuracy_Change_vs_Baseline_pp"] = (final_acc - baseline_acc) * 100
        if baseline_acc != 0: df.loc[index, "Accuracy_Retention_Percent"] = (final_acc / baseline_acc) * 100
        else: df.loc[index, "Accuracy_Retention_Percent"] = pd.NA


    model_size_disk = pd.to_numeric(row_series.get("Model_Size_MB_Disk"), errors='coerce')
    if pd.notna(model_size_disk) and pd.notna(baseline_size_disk) and baseline_size_disk != 0:
        df.loc[index, "Model_Size_Reduction_vs_Baseline_Percent"] = ((baseline_size_disk - model_size_disk) / baseline_size_disk) * 100
    else: df.loc[index, "Model_Size_Reduction_vs_Baseline_Percent"] = pd.NA


    current_params = pd.to_numeric(row_series.get("Params_Millions"), errors='coerce')
    if pd.notna(current_params) and pd.notna(baseline_params) and baseline_params != 0:
        df.loc[index, "Params_Reduction_vs_Baseline_Percent"] = ((baseline_params - current_params) / baseline_params) * 100
    else: df.loc[index, "Params_Reduction_vs_Baseline_Percent"] = pd.NA

    current_flops = pd.to_numeric(row_series.get("FLOPs_GMACs"), errors='coerce')
    if pd.notna(current_flops) and pd.notna(baseline_flops) and baseline_flops != 0:
        df.loc[index, "FLOPs_Reduction_vs_Baseline_Percent"] = ((baseline_flops - current_flops) / baseline_flops) * 100
    else: df.loc[index, "FLOPs_Reduction_vs_Baseline_Percent"] = pd.NA

    infer_cpu = pd.to_numeric(row_series.get("Inference_Time_ms_CPU (Batch 1)"), errors='coerce')
    if pd.notna(infer_cpu) and pd.notna(baseline_infer_cpu) and infer_cpu != 0:
        df.loc[index, "Inference_Speedup_vs_Baseline_CPU"] = baseline_infer_cpu / infer_cpu
    else: df.loc[index, "Inference_Speedup_vs_Baseline_CPU"] = pd.NA


    if torch.cuda.is_available() and "Baseline_Inference_Time_ms_GPU" in df.columns :
        baseline_infer_gpu = pd.to_numeric(current_baseline.get("inference_gpu_ms"), errors='coerce')
        df.loc[index, "Baseline_Inference_Time_ms_GPU"] = baseline_infer_gpu
        infer_gpu = pd.to_numeric(row_series.get("Inference_Time_ms_GPU (Batch 1)"), errors='coerce')
        if pd.notna(infer_gpu) and pd.notna(baseline_infer_gpu) and infer_gpu != 0:
            df.loc[index, "Inference_Speedup_vs_Baseline_GPU"] = baseline_infer_gpu / infer_gpu
        else: df.loc[index, "Inference_Speedup_vs_Baseline_GPU"] = pd.NA


# --- Save CSV (Identical to Script 2) ---
df.to_csv(OUTPUT_CSV, index=False, lineterminator='\n', float_format='%.5f')
print(f"\n--- Summary saved to {OUTPUT_CSV} ---")
if not df.empty:
    print("\nFirst 5 rows of the summary:")
    print(df.head().to_string())
else:
    print("DataFrame is empty.")
print(f"\nTotal experiments processed: {len(df)}")

print("--- Hybrid Script Finished ---")