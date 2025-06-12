import os
import json
import pandas as pd
import torch
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

print("--- Script Starting: Imports completed ---")

# --- Configuration ---
ROOT_DIR = "saved_models_and_logs"
OUTPUT_CSV = "new columns.csv"
DEFAULT_NUM_CLASSES = 1000 

# --- Uniform Evaluation Configuration ---
VALIDATION_DATA_PATH = "imagenet-mini/val" # <--- !!! ENSURE THIS PATH IS CORRECT !!!
BATCH_SIZE_EVAL = 32
NUM_WORKERS_EVAL = 0 
MAX_EVAL_BATCHES = 125 # SET LOW FOR TESTING, HIGH (e.g., 125 or float('inf')) FOR FINAL ROBUST ACCURACY

# For inference timing (using a single batch)
INPUT_TENSOR_CPU = torch.randn(1, 3, 224, 224)
# print("DEBUG: INPUT_TENSOR_CPU created.") # Optional debug
INPUT_TENSOR_GPU = None
if torch.cuda.is_available():
    # print("DEBUG: CUDA available, attempting to move input tensor to GPU...") # Optional debug
    try:
        INPUT_TENSOR_GPU = INPUT_TENSOR_CPU.cuda()
        # print("DEBUG: INPUT_TENSOR_GPU created on CUDA successfully.") # Optional debug
    except Exception as e_cuda_init:
        print(f"ERROR initializing INPUT_TENSOR_GPU on CUDA: {e_cuda_init}")
# else:
    # print("DEBUG: CUDA not available.") # Optional debug

WARMUP_INFERENCES = 2 
TIMED_INFERENCES = 5  

GPU_UNSTABLE_QUANTIZED_MODELS = [
    "resnet18pretrained_distilled_quant_ptq_int8_perchannel_post",
    "resnet18pretrained_distilled_quant_ptq_int8_pertensor_post",
    "resnet18pretrained_distilled_quant_qat_int8_epochs8",
    "resnet50_quant_ptq_int8_perchannel_post",
    "resnet50_quant_ptq_int8_pertensor_post",
    "resnet50_quant_qat_int8_epochs8",
]

# --- Helper: Image Transforms ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
eval_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,
])
# print("DEBUG: Evaluation transforms defined.") # Optional debug

current_eval_experiment_id = "" # Global for logging

# --- Helper: Uniform Evaluation ---
@torch.no_grad()
def evaluate_model_uniformly(model, device_str, num_classes_eval, max_batches_to_eval):
    global current_eval_experiment_id
    if not os.path.exists(VALIDATION_DATA_PATH):
        print(f"ERROR ({current_eval_experiment_id}): Val data path not found: {VALIDATION_DATA_PATH}")
        return "N/A (Val Data Missing)"
    try:
        val_dataset = ImageFolder(VALIDATION_DATA_PATH, eval_transforms)
        if len(val_dataset.classes) != num_classes_eval:
            print(f"WARNING ({current_eval_experiment_id}): Dataset classes ({len(val_dataset.classes)}) vs Model classes ({num_classes_eval}). Accuracy may be misleading.")
        if len(val_dataset) == 0:
            print(f"WARNING ({current_eval_experiment_id}): Validation dataset at '{VALIDATION_DATA_PATH}' is empty.")
            return 0.0 
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False, 
                                num_workers=NUM_WORKERS_EVAL, pin_memory=True if device_str=='cuda' else False)
    except Exception as e:
        print(f"ERROR ({current_eval_experiment_id}): Could not load validation data: {e}")
        return f"N/A (Val Data Load Error: {str(e).splitlines()[0]})"
    device = torch.device(device_str)
    model.to(device); model.eval()
    correct = 0; total = 0; batches_processed = 0
    if max_batches_to_eval == float('inf'): 
        print(f"      INFO ({current_eval_experiment_id}): Evaluating on ALL batches on device {device_str}.")
    else:
        print(f"      INFO ({current_eval_experiment_id}): Evaluating on device {device_str} for max {max_batches_to_eval} batches.")

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0); correct += (predicted == labels).sum().item()
        batches_processed += 1
        if batches_processed >= max_batches_to_eval: break
    return (correct / total) if total > 0 else 0.0

# --- Helper Functions: Model File and Size ---
def get_model_file_path(experiment_path):
    pth_files = list(Path(experiment_path).glob("*.pth"))
    if pth_files:
        for common_name in ["model_final.pth", "model_quantized.pth"]: # JIT often saved as model_quantized.pth
            for p_file in pth_files:
                if p_file.name == common_name: return p_file
        for p_file in pth_files: # Specific baseline names
            if "baseline_ft_imagenetmini_final.pth" in p_file.name: return p_file
        return pth_files[0]
    return None

def get_model_size_mb(model_path):
    if model_path and os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return None

# --- Core Function: Process Single Model ---
def process_single_model_file(model_path, base_arch, num_classes_model, experiment_id_str, config_details_dict):
    global current_eval_experiment_id, baseline_metrics # Make baseline_metrics accessible
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
    model_load_error_msg_specific = None 
    load_method_used = "Unknown" # For debugging JIT models

    try:
        _loaded_content = None 
        try:
            loaded_model_obj = torch.jit.load(model_path, map_location='cpu')
            load_method_used = "torch.jit.load()"
        except Exception: pass 
        
        if loaded_model_obj is None: 
            try:
                _loaded_full_model = torch.load(model_path, map_location='cpu', weights_only=False) 
                if isinstance(_loaded_full_model, torch.nn.Module):
                    loaded_model_obj = _loaded_full_model
                    load_method_used = "torch.load() [full model]"
                elif isinstance(_loaded_full_model, dict): 
                    _loaded_content = _loaded_full_model # Pass to state_dict loading
                    load_method_used = "torch.load() [content is dict]"
                else: 
                    model_load_error_msg_specific = f"N/A (Loaded obj not nn.Module/dict: {type(_loaded_full_model)})"
                    raise ValueError(model_load_error_msg_specific)
            except Exception as e_full_load:  # Catch if torch.load itself fails (e.g. corrupted file)
                 if loaded_model_obj is None and _loaded_content is None: 
                    try: # Last attempt to see if it's a dict if previous load failed entirely
                        _loaded_content = torch.load(model_path, map_location='cpu', weights_only=False)
                        if not isinstance(_loaded_content, dict): # If still not dict, then it's a real problem
                             raise e_full_load # Re-raise original error
                        load_method_used = "torch.load() [content is dict, after initial fail]"
                    except Exception as e_final_load: # If even this fails
                         model_load_error_msg_specific = f"N/A (File Load Fail: {e_final_load})"
                         raise ValueError(model_load_error_msg_specific)

        if loaded_model_obj is None: 
            if not isinstance(_loaded_content, dict): 
                 model_load_error_msg_specific = "N/A (File not JIT/Full/StateDict after attempts)"
                 raise ValueError(model_load_error_msg_specific)
            
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
            
            model_instance.load_state_dict(state_dict_to_load) # This is where structured pruning fails
            loaded_model_obj = model_instance
            load_method_used = "torch.load() [state_dict]"
        
        if loaded_model_obj is None: raise ValueError("Model not loaded after all attempts")
        loaded_model_obj.eval() 
        # print(f"      DEBUG ({experiment_id_str}): Model loaded using {load_method_used}.")

    except Exception as e:
        error_detail = str(e).splitlines()[0]
        final_error_msg = model_load_error_msg_specific if model_load_error_msg_specific else f"N/A (Model Load Error: {error_detail})"
        if "Error(s) in loading state_dict for ResNet" in str(e): # Specific for structured pruning
            final_error_msg = "N/A (Arch mismatch/StateDict load fail)"
        elif "Unsupported qscheme" in str(e) or "SerializedAttributes" in str(e): 
            final_error_msg = f"N/A (Quantized Load Err: {error_detail})"
        for key in results: results[key] = final_error_msg;
        if torch.cuda.is_available() and final_error_msg.startswith("N/A ("): results['Inference_Time_ms_GPU (Batch 1)'] = final_error_msg
        return results 

    # --- FLOPs and Params Calculation ---
    flops_calculated_by_thop = False
    # Only attempt thop if loaded_model_obj is a standard nn.Module, not a JIT ScriptModule
    if isinstance(loaded_model_obj, torch.nn.Module) and not isinstance(loaded_model_obj, torch.jit.ScriptModule):
        try:
            model_for_flops_params = loaded_model_obj.to(torch.device('cpu')) 
            dummy_input_flops = torch.randn(1, 3, 224, 224).to(torch.device('cpu'))
            macs, params = profile(model_for_flops_params, inputs=(dummy_input_flops,), verbose=False)
            results['FLOPs_GMACs'] = macs / 1e9
            results['Params_Millions'] = params / 1e6
            flops_calculated_by_thop = True
            del model_for_flops_params
        except Exception as e_flops:
            # print(f"      WARNING ({experiment_id_str}): Could not calculate FLOPs/Params with thop: {e_flops}")
            pass # Keep N/A, will try fallback
    elif isinstance(loaded_model_obj, torch.jit.ScriptModule):
         print(f"      INFO ({experiment_id_str}): Skipping direct thop for JIT ScriptModule. Will use baseline if applicable.")
    
    is_ao_quant_or_kmeans = "ptq" in experiment_id_str.lower() or \
                            "qat" in experiment_id_str.lower() or \
                            "kmeans" in experiment_id_str.lower()

    if not flops_calculated_by_thop and (is_ao_quant_or_kmeans or isinstance(loaded_model_obj, torch.jit.ScriptModule)):
        if base_arch in baseline_metrics and baseline_metrics[base_arch]:
            # print(f"      INFO ({experiment_id_str}): Using baseline {base_arch} FLOPs/Params as thop failed/skipped for quantized/JIT model.")
            baseline_f = baseline_metrics[base_arch].get("flops_gmacs")
            baseline_p = baseline_metrics[base_arch].get("params_millions")
            if pd.notna(baseline_f): results['FLOPs_GMACs'] = baseline_f
            else: results['FLOPs_GMACs'] = "N/A (Baseline FLOPs Missing)"
            if pd.notna(baseline_p): results['Params_Millions'] = baseline_p
            else: results['Params_Millions'] = "N/A (Baseline Params Missing)"
        else:
            # print(f"      WARNING ({experiment_id_str}): Baseline {base_arch} metrics not found for FLOPs/Params fallback.")
            results['FLOPs_GMACs'] = "N/A (Thop Skip/Baseline Miss)" # More specific N/A
            results['Params_Millions'] = "N/A (Thop Skip/Baseline Miss)"
    elif not flops_calculated_by_thop: 
        results['FLOPs_GMACs'] = "N/A (FLOPs Error)"
        results['Params_Millions'] = "N/A (Params Error)"
    
    # --- Uniform Accuracy Evaluation ---
    is_gpu_unstable_model = experiment_id_str in GPU_UNSTABLE_QUANTIZED_MODELS
    eval_device_for_acc = 'cpu' if is_gpu_unstable_model else ('cuda' if torch.cuda.is_available() else 'cpu')
    if is_gpu_unstable_model: 
        print(f"      INFO ({experiment_id_str}): Known GPU unstable. Forcing CPU evaluation for accuracy.")
    accuracy_val = evaluate_model_uniformly(loaded_model_obj, eval_device_for_acc, num_classes_model, MAX_EVAL_BATCHES)
    results['Final_Val_Accuracy_Uniform'] = accuracy_val

    # --- CPU Inference Timing ---
    cpu_model_for_timing = loaded_model_obj.to(torch.device('cpu')) 
    try:
        with torch.no_grad():
            for _ in range(WARMUP_INFERENCES): _ = cpu_model_for_timing(INPUT_TENSOR_CPU)
            timings_cpu = []
            for _ in range(TIMED_INFERENCES):
                start_time = time.perf_counter(); _ = cpu_model_for_timing(INPUT_TENSOR_CPU); end_time = time.perf_counter()
                timings_cpu.append((end_time - start_time) * 1000)
        results['Inference_Time_ms_CPU (Batch 1)'] = sum(timings_cpu) / len(timings_cpu) if timings_cpu else "N/A (CPU Time Error)"
    except Exception as e_cpu_time: results['Inference_Time_ms_CPU (Batch 1)'] = f"N/A (CPU Time Error: {str(e_cpu_time).splitlines()[0]})"

    # --- GPU Inference Timing ---
    if is_gpu_unstable_model:
        results['Inference_Time_ms_GPU (Batch 1)'] = "N/A (Known JIT GPU Unstable)"
    elif torch.cuda.is_available() and INPUT_TENSOR_GPU is not None:
        gpu_model_for_timing = loaded_model_obj.to(torch.device('cuda'))
        try:
            with torch.no_grad():
                for _ in range(WARMUP_INFERENCES): _ = gpu_model_for_timing(INPUT_TENSOR_GPU); torch.cuda.synchronize()
                timings_gpu = []
                for _ in range(TIMED_INFERENCES):
                    torch.cuda.synchronize(); start_time = time.perf_counter(); _ = gpu_model_for_timing(INPUT_TENSOR_GPU); torch.cuda.synchronize(); end_time = time.perf_counter()
                    timings_gpu.append((end_time - start_time) * 1000)
            results['Inference_Time_ms_GPU (Batch 1)'] = sum(timings_gpu) / len(timings_gpu) if timings_gpu else "N/A (GPU Time Error)"
        except Exception as e_gpu_time: results['Inference_Time_ms_GPU (Batch 1)'] = f"N/A (GPU Time Error: {str(e_gpu_time).splitlines()[0]})"
        if 'gpu_model_for_timing' in locals(): del gpu_model_for_timing
    elif not torch.cuda.is_available(): results['Inference_Time_ms_GPU (Batch 1)'] = "N/A (CUDA unavailable)"
    
    del loaded_model_obj
    if 'cpu_model_for_timing' in locals(): del cpu_model_for_timing
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect() 
    return results

# --- Main Processing Logic (Loops & CSV generation) ---
# (This part remains identical to your last version where you copied it)
# Ensure your main loops, desired_columns, and relative metrics calculations are complete and correct.
# I will put placeholders here for brevity, assuming you have this part from the previous script.

# --- Start of Main Script Execution ---
all_experiments_data = []
baseline_metrics = {} 
print(f"DEBUG: Checking VALIDATION_DATA_PATH: {VALIDATION_DATA_PATH}")
if not os.path.exists(VALIDATION_DATA_PATH): print(f"FATAL ERROR: VALIDATION_DATA_PATH '{VALIDATION_DATA_PATH}' does not exist."); exit()
if not os.listdir(VALIDATION_DATA_PATH) : print(f"FATAL ERROR: VALIDATION_DATA_PATH '{VALIDATION_DATA_PATH}' exists but is empty or not readable."); exit()
print("DEBUG: VALIDATION_DATA_PATH check passed.")
print("\n--- Processing baselines ---")
for cat_name_outer in os.listdir(ROOT_DIR):
    cat_path_outer = os.path.join(ROOT_DIR, cat_name_outer)
    if os.path.isdir(cat_path_outer) and ("baseline" in cat_name_outer.lower()):
        exp_name = cat_name_outer ; print(f"Processing baseline experiment: {exp_name}"); exp_path = cat_path_outer; row = {"Experiment_ID": exp_name}
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
        measured_metrics = process_single_model_file(model_file_path, row["Base_Model_Arch"], num_classes_model, exp_name, config_details)
        row.update(measured_metrics) # Update row with all measured metrics
        if row["Base_Model_Arch"] != "Unknown":
            baseline_metrics[row["Base_Model_Arch"]] = {"val_accuracy": pd.to_numeric(row["Final_Val_Accuracy_Uniform"], errors='coerce'),"model_size_mb_disk": pd.to_numeric(row["Model_Size_MB_Disk"], errors='coerce'),"inference_cpu_ms": pd.to_numeric(row["Inference_Time_ms_CPU (Batch 1)"], errors='coerce'),"inference_gpu_ms": pd.to_numeric(row["Inference_Time_ms_GPU (Batch 1)"], errors='coerce'), "flops_gmacs": pd.to_numeric(row["FLOPs_GMACs"], errors='coerce'), "params_millions": pd.to_numeric(row["Params_Millions"], errors='coerce')}
        all_experiments_data.append(row)

print("\n--- Processing other experiments ---")
for cat_name in os.listdir(ROOT_DIR):
    cat_path = os.path.join(ROOT_DIR, cat_name)
    if not os.path.isdir(cat_path) or "baseline" in cat_name.lower(): continue 
    print(f"Processing category: {cat_name}")
    for exp_name in os.listdir(cat_path):
        exp_path = os.path.join(cat_path, exp_name)
        if not os.path.isdir(exp_path): continue
        print(f"  Processing experiment: {exp_name}"); row = {"Experiment_ID": exp_name}
        if "resnet18" in exp_name.lower(): row["Base_Model_Arch"] = "ResNet18"
        elif "resnet50" in exp_name.lower(): row["Base_Model_Arch"] = "ResNet50"
        else: 
            if cat_name == "combined_distilled_quantized": row["Base_Model_Arch"] = "ResNet18"
            else: row["Base_Model_Arch"] = "ResNet50"
        opt_cat_map = {"combined_distilled_quantized": "Combined", "knowledge_distillation": "Knowledge Distillation","pruning_nm_sparsity": "Pruning", "pruning_structured_iterative": "Pruning","pruning_structured_oneshot": "Pruning", "pruning_unstructured_iterative": "Pruning","pruning_unstructured_oneshot": "Pruning", "quantization_kmeans": "Quantization","quantization_ptq_int8": "Quantization", "quantization_qat_int8": "Quantization",}
        row["Optimization_Category"] = opt_cat_map.get(cat_name, "Other")
        log_path = os.path.join(exp_path, "log.json"); model_file_path = get_model_file_path(exp_path)
        log_data, config_details, training_summary, original_eval_metrics, quant_specific_details = {}, {}, {}, {}, {}
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f: log_data = json.load(f)
            except json.JSONDecodeError:
                print(f"    Error decoding JSON for {log_path}")
                for key_to_fill in ["Specific_Technique", "Key_Parameters", "Final_Val_Accuracy_Uniform", "Model_Size_MB_Disk", "Inference_Time_ms_CPU (Batch 1)", "FLOPs_GMACs", "Params_Millions"]: row[key_to_fill] = "JSON Error"
                all_experiments_data.append(row); continue
        config_details = log_data.get('config_details', {}); training_summary = log_data.get('training_summary', {}); original_eval_metrics = log_data.get('original_evaluation_metrics_from_log', {}); quant_specific_details = log_data.get('quantization_specific_details', {})
        specific_tech_parts, key_params_parts = [], []
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
        if "prune_nm" in pruning_tech_exp_name or "nm_sparsity" in pruning_method_cfg:
            if "N:M Sparsity" not in specific_tech_parts: specific_tech_parts.append("N:M Sparsity")
            n = config_details.get('nm_sparsity_n', 2); m = config_details.get('nm_sparsity_m', 4); key_params_parts.append(f"N:{n}, M:{m}")
        elif "prune_struct_it" in pruning_tech_exp_name or "iterative_structured" in pruning_strat_cfg: specific_tech_parts.append("Iterative Structured Pruning (L1 Filter)")
        elif "prune_struct_os" in pruning_tech_exp_name or "one_shot_structured" in pruning_strat_cfg or "structured_l1_filter" in pruning_method_cfg : specific_tech_parts.append("One-Shot Structured Pruning (L1 Filter)")
        elif "prune_unstruct_it" in pruning_tech_exp_name or "iterative_unstructured" in pruning_strat_cfg : specific_tech_parts.append("Iterative Unstructured Pruning (L1)")
        elif "prune_unstruct_os" in pruning_tech_exp_name or "one_shot_unstructured" in pruning_strat_cfg : specific_tech_parts.append("One-Shot Unstructured Pruning (L1)")
        if any("Pruning" in tech for tech in specific_tech_parts):
            target_sparsities = [config_details.get('target_overall_sparsity_approx_for_this_stage'), config_details.get('target_filter_pruning_rate_per_layer'), config_details.get('target_sparsity_for_this_stage'), config_details.get('target_sparsity')]
            for sp_val in target_sparsities:
                if sp_val is not None:
                    try: key_params_parts.append(f"Target Sparsity: {float(sp_val)*100:.1f}%")
                    except ValueError: key_params_parts.append(f"Target Sparsity: {sp_val}")
                    break
        row["Specific_Technique"] = " + ".join(list(dict.fromkeys(specific_tech_parts))) if specific_tech_parts else "Other"; row["Key_Parameters"] = "; ".join(key_params_parts) if key_params_parts else "N/A"
        row["Accuracy_Drop_From_Best_Epoch_pp"] = "N/A"
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
        if 'student_config' in config_details and isinstance(config_details['student_config'], dict): num_classes_model = config_details['student_config'].get('num_classes', num_classes_model)
        measured_metrics = process_single_model_file(model_file_path, row["Base_Model_Arch"], num_classes_model, exp_name, config_details)
        row.update(measured_metrics) # Update row with all measured metrics
        all_experiments_data.append(row)

# 3. Create DataFrame & Finalize Columns
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
for col in desired_columns:
    if col not in df.columns: df[col] = pd.NA 
df = df[desired_columns]

# 4. Calculate Relative Metrics 
print("\n--- Calculating relative metrics ---")
for index, row_series in df.iterrows():
    baseline_arch_to_use = "ResNet50" 
    if str(row_series.get("Optimization_Category","")).strip() in ["Knowledge Distillation", "Combined"]: baseline_arch_to_use = "ResNet18"
    elif "resnet18" in str(row_series.get("Experiment_ID","")).lower() and row_series.get("Base_Model_Arch") == "ResNet18": baseline_arch_to_use = "ResNet18"
    if row_series.get("Optimization_Category") == "Baseline":
        df.loc[index, "Baseline_Val_Accuracy"] = pd.to_numeric(row_series.get("Final_Val_Accuracy"), errors='coerce')
        df.loc[index, "Accuracy_Change_vs_Baseline_pp"] = 0.0; df.loc[index, "Accuracy_Retention_Percent"] = 100.0
        df.loc[index, "Baseline_Model_Size_MB_Disk"] = pd.to_numeric(row_series.get("Model_Size_MB_Disk"), errors='coerce')
        df.loc[index, "Model_Size_Reduction_vs_Baseline_Percent"] = 0.0
        df.loc[index, "Baseline_Params_Millions"] = pd.to_numeric(row_series.get("Params_Millions"), errors='coerce') # For baseline itself
        df.loc[index, "Params_Reduction_vs_Baseline_Percent"] = 0.0
        df.loc[index, "Baseline_FLOPs_GMACs"] = pd.to_numeric(row_series.get("FLOPs_GMACs"), errors='coerce') # For baseline itself
        df.loc[index, "FLOPs_Reduction_vs_Baseline_Percent"] = 0.0
        df.loc[index, "Baseline_Inference_Time_ms_CPU"] = pd.to_numeric(row_series.get("Inference_Time_ms_CPU (Batch 1)"), errors='coerce')
        df.loc[index, "Inference_Speedup_vs_Baseline_CPU"] = 1.0
        if torch.cuda.is_available() and "Baseline_Inference_Time_ms_GPU" in df.columns:
            df.loc[index, "Baseline_Inference_Time_ms_GPU"] = pd.to_numeric(row_series.get("Inference_Time_ms_GPU (Batch 1)"), errors='coerce')
            df.loc[index, "Inference_Speedup_vs_Baseline_GPU"] = 1.0
        continue
    if baseline_arch_to_use not in baseline_metrics or not baseline_metrics[baseline_arch_to_use]: continue
    current_baseline = baseline_metrics[baseline_arch_to_use]
    baseline_acc = pd.to_numeric(current_baseline.get("val_accuracy"), errors='coerce'); baseline_size_disk = pd.to_numeric(current_baseline.get("model_size_mb_disk"), errors='coerce'); baseline_params = pd.to_numeric(current_baseline.get("params_millions"), errors='coerce'); baseline_flops = pd.to_numeric(current_baseline.get("flops_gmacs"), errors='coerce'); baseline_infer_cpu = pd.to_numeric(current_baseline.get("inference_cpu_ms"), errors='coerce')
    df.loc[index, "Baseline_Val_Accuracy"] = baseline_acc; df.loc[index, "Baseline_Model_Size_MB_Disk"] = baseline_size_disk; df.loc[index, "Baseline_Params_Millions"] = baseline_params; df.loc[index, "Baseline_FLOPs_GMACs"] = baseline_flops; df.loc[index, "Baseline_Inference_Time_ms_CPU"] = baseline_infer_cpu
    final_acc = pd.to_numeric(row_series.get("Final_Val_Accuracy"), errors='coerce')
    if pd.notna(final_acc) and pd.notna(baseline_acc):
        df.loc[index, "Accuracy_Change_vs_Baseline_pp"] = (final_acc - baseline_acc) * 100
        if baseline_acc != 0: df.loc[index, "Accuracy_Retention_Percent"] = (final_acc / baseline_acc) * 100
    model_size_disk = pd.to_numeric(row_series.get("Model_Size_MB_Disk"), errors='coerce')
    if pd.notna(model_size_disk) and pd.notna(baseline_size_disk) and baseline_size_disk != 0: df.loc[index, "Model_Size_Reduction_vs_Baseline_Percent"] = ((baseline_size_disk - model_size_disk) / baseline_size_disk) * 100
    current_params = pd.to_numeric(row_series.get("Params_Millions"), errors='coerce')
    if pd.notna(current_params) and pd.notna(baseline_params) and baseline_params != 0: df.loc[index, "Params_Reduction_vs_Baseline_Percent"] = ((baseline_params - current_params) / baseline_params) * 100
    current_flops = pd.to_numeric(row_series.get("FLOPs_GMACs"), errors='coerce')
    if pd.notna(current_flops) and pd.notna(baseline_flops) and baseline_flops != 0: df.loc[index, "FLOPs_Reduction_vs_Baseline_Percent"] = ((baseline_flops - current_flops) / baseline_flops) * 100
    infer_cpu = pd.to_numeric(row_series.get("Inference_Time_ms_CPU (Batch 1)"), errors='coerce')
    if pd.notna(infer_cpu) and pd.notna(baseline_infer_cpu) and infer_cpu != 0: df.loc[index, "Inference_Speedup_vs_Baseline_CPU"] = baseline_infer_cpu / infer_cpu
    if torch.cuda.is_available() and "Baseline_Inference_Time_ms_GPU" in df.columns :
        baseline_infer_gpu = pd.to_numeric(current_baseline.get("inference_gpu_ms"), errors='coerce')
        df.loc[index, "Baseline_Inference_Time_ms_GPU"] = baseline_infer_gpu
        infer_gpu = pd.to_numeric(row_series.get("Inference_Time_ms_GPU (Batch 1)"), errors='coerce')
        if pd.notna(infer_gpu) and pd.notna(baseline_infer_gpu) and infer_gpu != 0: df.loc[index, "Inference_Speedup_vs_Baseline_GPU"] = baseline_infer_gpu / infer_gpu

# 5. Save CSV
df.to_csv(OUTPUT_CSV, index=False, lineterminator='\n', float_format='%.5f')
print(f"\n--- Summary saved to {OUTPUT_CSV} ---")
print("\nFirst 5 rows of the summary:")
print(df.head().to_string())
print(f"\nTotal experiments processed: {len(df)}")

print("--- Script Finished ---")