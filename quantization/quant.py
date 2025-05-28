import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.ao.quantization as ao_quant # Use ao.quantization
from torch.ao.quantization import QConfig, MinMaxObserver, PerChannelMinMaxObserver
import torchvision.models.quantization as q_models # For quantizable ResNet

import json
import os
import time
import copy
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import traceback # For better error printing

# For dummy data creation if PIL is available
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def get_model_size(model_path):
    try:
        return os.path.getsize(model_path) / (1024 * 1024)
    except FileNotFoundError:
        print(f"Warning: File not found at {model_path} for size check.")
        return 0

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    non_zero = 0
    for p in model.parameters():
        is_quantized_param = hasattr(p, 'is_quantized') and p.is_quantized
        if is_quantized_param:
            try:
                # For modules quantized with torch.ao.quantization
                if hasattr(p, 'int_repr'):
                    int_data = p.int_repr()
                    non_zero += (int_data != 0).sum().item()
                elif hasattr(p, '_packed_params') and hasattr(p._packed_params, 'unpack'): # e.g. LinearPackedParams
                    # This path is more complex, as _packed_params might hold weight and bias
                    # For simplicity here, if int_repr isn't available, we might approximate
                    # or accept that non_zero count for these specific packed formats might be total.
                    # A more robust way would be to unpack and count.
                    # For now, let's assume typical quantized layers have int_repr or are handled by fallback.
                    non_zero += (p.data !=0).sum().item() # Fallback for now
                else:
                    non_zero += (p.data != 0).sum().item()
            except Exception: non_zero += (p.data !=0).sum().item()
        elif p.data.is_floating_point(): non_zero += (p.data != 0).sum().item()
        else: non_zero += (p.data != 0).sum().item() # For non-quantized, non-float (e.g. int params if any)
    return total, non_zero

def measure_inference_speed(model, data_loader_bs1, device, num_samples_to_measure=1000, warmup_iterations=20):
    """
    Measure inference speed for a specific number of samples, mimicking pruning script's method.
    - Uses dummy data for warm-up.
    - data_loader_bs1 should have batch_size=1, be derived from the full validation set,
      and have pin_memory=True if device is CUDA.
    """
    model.eval()
    model.to(device)
    total_time = 0
    samples_processed = 0

    # print(f"  [measure_inference_speed] Using data_loader with {len(data_loader_bs1.dataset)} total items (bs=1).")

    # Warm-up phase (using dummy data like pruning script)
    if warmup_iterations > 0:
        print(f"    Warm-up for inference ({warmup_iterations} iterations) on {device} using dummy data...")
        # Create dummy input on the target device to avoid H2D copy during warmup loop
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        for _ in range(warmup_iterations):
            with torch.no_grad(): # Ensure no_grad during warmup model calls
                _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    print(f"    Measuring inference speed on {device} for up to {num_samples_to_measure} samples...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader_bs1): # data_loader_bs1 has batch_size=1
            if samples_processed >= num_samples_to_measure:
                break
            
            data = data.to(device) # current_batch_size is 1
            
            start_time = time.perf_counter()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            total_time += (end_time - start_time)
            # In batch_size=1, data.size(0) is 1.
            # Using data.size(0) is more robust if bs could change, but here it's fixed.
            samples_processed += data.size(0)

    if samples_processed == 0:
        print("    Warning: No samples processed during inference speed measurement.")
        return {"images_per_second": 0, "latency_ms_per_image": float('inf'),
                "total_images_measured": 0, "total_time_seconds": 0}

    avg_time_per_sample = total_time / samples_processed
    images_per_second = 1.0 / avg_time_per_sample
    
    return {"images_per_second": images_per_second,
            "latency_ms_per_image": avg_time_per_sample * 1000,
            "total_images_measured": samples_processed,
            "total_time_seconds": total_time}

def evaluate_model(model, data_loader, criterion, device, quick_check_name=""):
    """Evaluate model accuracy and loss"""
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * data.size(0) 
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    return accuracy, avg_loss

def calibrate_model(model, data_loader, device, num_batches_to_use=100):
    model.eval()
    model.to(device)
    print(f"  Calibrating model on {device} with up to {num_batches_to_use} batches (loader has {len(data_loader)} batches)...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= num_batches_to_use:
                break
            data = data.to(device)
            model(data)
    print("  Calibration complete.")

def apply_ptq_per_tensor(
    original_fp32_model_state_dict, num_classes, calib_loader, ptq_device, global_config):
    print("  Applying PTQ (per-tensor)...")
    model_to_quantize = q_models.resnet50(weights=None, quantize=False)
    model_to_quantize.fc = nn.Linear(model_to_quantize.fc.in_features, num_classes)
    model_to_quantize.load_state_dict(original_fp32_model_state_dict)
    model_to_quantize.to(ptq_device); model_to_quantize.eval()
    print("    Fusing model modules...")
    model_to_quantize.fuse_model(is_qat=False)
    model_to_quantize.qconfig = ao_quant.get_default_qconfig(global_config.get('ptq_backend', 'fbgemm'))
    print("    Preparing model for static quantization...")
    ao_quant.prepare(model_to_quantize, inplace=True)
    calibrate_model(model_to_quantize, calib_loader, ptq_device,
                    num_batches_to_use=global_config.get('ptq_calib_batches', 100))
    print("    Converting model to quantized version...")
    ao_quant.convert(model_to_quantize, inplace=True)
    return model_to_quantize

def apply_ptq_per_channel_manual(
    original_fp32_quantizable_model_sd, num_classes, train_loader_for_calib, ptq_device, global_config):
    print("  Applying PTQ (per-channel manually defined)...")
    model_to_quantize = q_models.resnet50(weights=None, quantize=False)
    model_to_quantize.fc = nn.Linear(model_to_quantize.fc.in_features, num_classes)
    model_to_quantize.load_state_dict(original_fp32_quantizable_model_sd)
    model_to_quantize.to(ptq_device); model_to_quantize.eval()
    backend = global_config.get('ptq_backend', 'fbgemm')
    if backend == 'fbgemm':
        qscheme_weight = torch.per_channel_symmetric; activation_reduce_range = False; weight_reduce_range = False
    elif backend == 'qnnpack':
        qscheme_weight = torch.per_channel_affine; activation_reduce_range = True; weight_reduce_range = False
    else:
        print(f"    Warning: Unknown PTQ backend '{backend}'. Using fbgemm-like defaults.")
        qscheme_weight = torch.per_channel_symmetric; activation_reduce_range = False; weight_reduce_range = False
    per_channel_qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=activation_reduce_range),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=qscheme_weight, reduce_range=weight_reduce_range, ch_axis=0))
    model_to_quantize.qconfig = per_channel_qconfig
    print(f"    Fusing model modules for backend '{backend}'...")
    model_to_quantize.fuse_model(is_qat=False)
    print("    Preparing model for static quantization...")
    ao_quant.prepare(model_to_quantize, inplace=True)
    calibrate_model(model_to_quantize, train_loader_for_calib, ptq_device, global_config.get('ptq_calib_batches', 100))
    print("    Converting model to quantized version...")
    ao_quant.convert(model_to_quantize, inplace=True)
    return model_to_quantize

def apply_qat(
    original_fp32_model_state_dict, num_classes, train_loader, val_loader, # val_loader here is quick_val_loader
    qat_training_device, global_config):
    print("  Applying QAT...")
    qat_model_fp32 = q_models.resnet50(weights=None, quantize=False)
    qat_model_fp32.fc = nn.Linear(qat_model_fp32.fc.in_features, num_classes)
    qat_model_fp32.load_state_dict(original_fp32_model_state_dict)
    qat_model_fp32.train() 
    qat_model_fp32.qconfig = ao_quant.get_default_qat_qconfig(global_config.get('qat_backend', 'fbgemm'))
    print("    Fusing model for QAT...")
    qat_model_fp32.fuse_model(is_qat=True)
    print("    Preparing model for QAT...")
    model_prepared = ao_quant.prepare_qat(qat_model_fp32, inplace=False) 
    model_prepared.to(qat_training_device)
    criterion = nn.CrossEntropyLoss(); optimizer = optim.SGD(model_prepared.parameters(), 
        lr=global_config['qat_learning_rate'], momentum=global_config['qat_momentum'], weight_decay=global_config['qat_weight_decay'])
    history = defaultdict(list); start_time = time.time()
    for epoch in range(global_config['qat_epochs']):
        model_prepared.train(); running_loss = 0.0; correct_train = 0; total_train = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(qat_training_device), targets.to(qat_training_device)
            optimizer.zero_grad(); outputs = model_prepared(data); loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * data.size(0); _, predicted = outputs.max(1)
            total_train += targets.size(0); correct_train += predicted.eq(targets).sum().item()
            if (batch_idx + 1) % global_config.get('qat_log_interval', 100) == 0:
                print(f'    QAT Epoch [{epoch+1}/{global_config["qat_epochs"]}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        epoch_loss = running_loss / total_train if total_train > 0 else 0
        epoch_acc = correct_train / total_train if total_train > 0 else 0
        val_acc, val_loss = evaluate_model(model_prepared, val_loader, criterion, qat_training_device, "QAT Epoch Val")
        history['loss'].append(epoch_loss); history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss); history['val_accuracy'].append(val_acc)
        print(f'    QAT Epoch [{epoch+1}] Train L: {epoch_loss:.4f} A: {epoch_acc:.4f} | Val L: {val_loss:.4f} A: {val_acc:.4f} (on quick_val_loader)')
    training_time = time.time() - start_time
    model_prepared.eval(); model_prepared_cpu = model_prepared.to(torch.device('cpu'))
    model_quantized = ao_quant.convert(model_prepared_cpu, inplace=False)
    return model_quantized, history, training_time

def apply_kmeans_quantization(original_fp32_model_cpu, n_clusters=256):
    print(f"  Applying K-means quantization with {n_clusters} clusters...")
    quantized_model = copy.deepcopy(original_fp32_model_cpu) 
    codebooks = {}
    for name, param in quantized_model.named_parameters():
        if 'weight' in name and param.dim() > 1 and param.requires_grad:
            original_shape = param.shape
            weights_flat = param.data.cpu().numpy().flatten().astype(np.float32)
            unique_weights = np.unique(weights_flat)
            current_n_clusters = min(n_clusters, len(unique_weights))
            if current_n_clusters < 2 :
                print(f"    Skipping K-means for {name}: not enough unique values ({len(unique_weights)}) for {current_n_clusters} clusters.")
                continue
            kmeans = KMeans(n_clusters=current_n_clusters, random_state=42, n_init='auto')
            kmeans.fit(weights_flat.reshape(-1, 1))
            centroids = kmeans.cluster_centers_.flatten(); labels = kmeans.labels_
            quantized_weights = centroids[labels].reshape(original_shape)
            param.data = torch.tensor(quantized_weights, dtype=param.dtype, device=param.device)
            codebooks[name] = {'centroids': centroids.tolist(), 'num_clusters': len(centroids)}
    return quantized_model, codebooks

def run_quantization_experiments(config):
    training_device = torch.device('cuda' if torch.cuda.is_available() and config.get("use_cuda_if_available", True) else 'cpu')
    eval_device = torch.device('cpu') # CPU for final PTQ model evaluation, PTQ conversion, and K-Means
    print(f"Using training device: {training_device}, PTQ/Quantized Model Eval device: {eval_device}")

    save_dir = config['save_dir']; os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    try:
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(config['data_dir'], 'train'), transform=transform)
        val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(config['data_dir'], 'val'), transform=transform)
        if not dataset.samples: raise FileNotFoundError("No training images found.")
        if not val_dataset.samples: raise FileNotFoundError("No validation images found.")
    except Exception as e: print(f"Error loading dataset: {e}. Check 'data_dir'."); traceback.print_exc(); return None
    num_classes = len(dataset.classes); print(f"Number of classes detected: {num_classes}")

    # Common loader parameters that will be expanded.
    # This dict will be used for QAT train, calib, and quick_val loaders.
    # Individual loaders can override shuffle. Batch size, num_workers, pin_memory come from here.
    # Pin memory is set based on training_device, assuming these loaders might be used
    # in contexts related to training or QAT validation on that device.
    common_loader_kwargs = {
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'pin_memory': (training_device.type == 'cuda')
    }

    calib_subset_indices = list(range(min(len(dataset), config.get('ptq_calib_dataset_size', 500))))
    calib_loader = DataLoader(
        torch.utils.data.Subset(dataset, calib_subset_indices),
        shuffle=False, # Specific to calib_loader
        **common_loader_kwargs
    )

    qat_train_loader = DataLoader(
        dataset,
        shuffle=True, # Specific to qat_train_loader
        **common_loader_kwargs
    )

    quick_check_val_samples = config.get("quick_check_val_samples", 200) # Adjusted from 1000 for quick ACCURACY checks
    quick_val_subset_indices = list(range(min(len(val_dataset), quick_check_val_samples)))
    quick_val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, quick_val_subset_indices),
        shuffle=False, # Specific to quick_val_loader
        **common_loader_kwargs # This provides batch_size, num_workers, pin_memory
    )
    # The above `quick_val_loader` definition fixes the TypeError.

    # Parameters for the "pruning-style" inference benchmark
    inference_benchmark_num_samples = config.get('inference_benchmark_samples', 1000)
    inference_benchmark_warmup_iters = config.get('inference_benchmark_warmup_iters', 20)

    # Create the DataLoader for the inference benchmark
    benchmark_loader_pin_memory = (training_device.type == 'cuda') # For GPU benchmarks
    benchmark_loader_num_workers = config['num_workers']

    print(f"Creating benchmark_inference_loader (bs=1) for full val_dataset ({len(val_dataset)} samples) with pin_memory={benchmark_loader_pin_memory}, num_workers={benchmark_loader_num_workers}")
    benchmark_inference_loader_bs1 = DataLoader(
        val_dataset,
        batch_size=1, # Explicitly 1 for this loader
        shuffle=False,
        num_workers=benchmark_loader_num_workers,
        pin_memory=benchmark_loader_pin_memory
    )

    print("Loading baseline FP32 model (standard torchvision.models.resnet50)...")
    baseline_fp32_model = torchvision.models.resnet50(weights=None)
    baseline_fp32_model.fc = nn.Linear(baseline_fp32_model.fc.in_features, num_classes)
    try:
        baseline_fp32_model.load_state_dict(torch.load(config['baseline_model_path'], map_location='cpu', weights_only=True))
        print(f"Loaded baseline model weights from {config['baseline_model_path']}")
    except Exception as e: print(f"Error loading baseline model: {e}"); traceback.print_exc(); return None
    criterion = nn.CrossEntropyLoss()
    results = {"experiment_type": "quantization", "config": config, "num_classes": num_classes, "quantization_runs": []}

    baseline_fp32_model_cpu = copy.deepcopy(baseline_fp32_model).to(eval_device) # eval_device is CPU
    print(f"\nEvaluating baseline FP32 model on CPU ({eval_device}) (accuracy with quick_val_loader)...")
    acc_cpu, loss_cpu = evaluate_model(baseline_fp32_model_cpu, quick_val_loader, criterion, eval_device, "Baseline CPU Acc")
    
    print(f"Measuring baseline FP32 model inference speed on CPU ({eval_device}) (with benchmark_inference_loader_bs1)...")
    infer_cpu = measure_inference_speed(baseline_fp32_model_cpu, benchmark_inference_loader_bs1, eval_device, 
                                        num_samples_to_measure=inference_benchmark_num_samples,
                                        warmup_iterations=inference_benchmark_warmup_iters)
    temp_path_cpu = os.path.join(save_dir, '_temp_baseline_cpu.pth'); torch.save(baseline_fp32_model_cpu.state_dict(), temp_path_cpu)
    size_cpu = get_model_size(temp_path_cpu); os.remove(temp_path_cpu)
    total_p_cpu, nz_p_cpu = count_parameters(baseline_fp32_model_cpu)
    results["baseline_fp32_cpu_benchmark"] = { # Renamed from _quick_check_cpu
        "run_type": "baseline_fp32_cpu_benchmark", "val_accuracy": acc_cpu, "val_loss": loss_cpu,
        "model_size_mb": size_cpu, "parameter_counts": {"total_params": total_p_cpu, "non_zero_params": nz_p_cpu},
        "inference_metrics": infer_cpu, "eval_samples_accuracy": len(quick_val_loader.dataset), "inf_samples_speed": infer_cpu['total_images_measured']}
    print(f"Baseline FP32 (CPU Benchmark): Acc: {acc_cpu:.4f}, Size: {size_cpu:.2f}MB, IPS: {infer_cpu['images_per_second']:.2f}")

    if training_device != eval_device: # If CUDA is available for training_device
        baseline_fp32_model_gpu = copy.deepcopy(baseline_fp32_model).to(training_device) # Use a fresh copy on GPU
        print(f"\nEvaluating baseline FP32 model on {training_device} (accuracy with quick_val_loader)...")
        acc_gpu, loss_gpu = evaluate_model(baseline_fp32_model_gpu, quick_val_loader, criterion, training_device, "Baseline GPU Acc")
        
        print(f"Measuring baseline FP32 model inference speed on {training_device} (with benchmark_inference_loader_bs1)...")
        infer_gpu = measure_inference_speed(baseline_fp32_model_gpu, benchmark_inference_loader_bs1, training_device,
                                            num_samples_to_measure=inference_benchmark_num_samples,
                                            warmup_iterations=inference_benchmark_warmup_iters)
        # Size and params are same as CPU version for FP32
        results["baseline_fp32_gpu_benchmark"] = { # Renamed from _quick_check_gpu
            "run_type": f"baseline_fp32_gpu_benchmark", "val_accuracy": acc_gpu, "val_loss": loss_gpu,
            "inference_metrics": infer_gpu, "eval_samples_accuracy": len(quick_val_loader.dataset), "inf_samples_speed": infer_gpu['total_images_measured']}
        print(f"Baseline FP32 ({training_device} Benchmark): Acc: {acc_gpu:.4f}, IPS: {infer_gpu['images_per_second']:.2f}")

    fp32_model_state_dict_for_quant = copy.deepcopy(baseline_fp32_model_cpu.state_dict())

    # Common args for model evaluation (accuracy on quick_val_loader, speed on benchmark_loader)
    # Note: Quantized models are typically evaluated on CPU (eval_device)
    common_eval_params_for_quantized_models = {
        "acc_loader": quick_val_loader, 
        "criterion": criterion, 
        "eval_device": eval_device, # PTQ/QAT models usually run on CPU for eval
        "speed_loader": benchmark_inference_loader_bs1,
        "num_inf_samples": inference_benchmark_num_samples,
        "warmup_inf_iters": inference_benchmark_warmup_iters
    }

    def record_quant_results(run_type, method_name, model_quant, model_path, training_info=None):
        # Ensure model_quant is on the correct device for evaluation
        model_quant.to(common_eval_params_for_quantized_models["eval_device"])
        model_quant.eval()

        print(f"  Evaluating {run_type} model accuracy on {common_eval_params_for_quantized_models['eval_device']}...")
        acc, loss = evaluate_model(model_quant, 
                                   common_eval_params_for_quantized_models["acc_loader"], 
                                   common_eval_params_for_quantized_models["criterion"], 
                                   common_eval_params_for_quantized_models["eval_device"], 
                                   run_type + " Acc")
        
        print(f"  Measuring {run_type} model inference speed on {common_eval_params_for_quantized_models['eval_device']}...")
        infer_metrics = measure_inference_speed(model_quant, 
                                                common_eval_params_for_quantized_models["speed_loader"], 
                                                common_eval_params_for_quantized_models["eval_device"],
                                                num_samples_to_measure=common_eval_params_for_quantized_models["num_inf_samples"],
                                                warmup_iterations=common_eval_params_for_quantized_models["warmup_inf_iters"])
        size = get_model_size(model_path); total_p, nz_p = count_parameters(model_quant)
        run_data = {
            "run_type": run_type, "quantization_method": method_name, "model_saved_as": model_path,
            "final_evaluation_metrics": {"val_accuracy": acc, "val_loss": loss, "model_size_mb": size,
            "parameter_counts": {"total_params": total_p, "non_zero_params": nz_p}, "inference_metrics": infer_metrics,
            "eval_samples_accuracy": len(common_eval_params_for_quantized_models["acc_loader"].dataset), 
            "inf_samples_speed": infer_metrics['total_images_measured']}}
        if training_info: run_data["training_config"] = training_info
        results["quantization_runs"].append(run_data)
        print(f"  {run_type}: Acc: {acc:.4f}, Size: {size:.2f}MB, IPS: {infer_metrics['images_per_second']:.2f} (on {common_eval_params_for_quantized_models['eval_device']})")

    if config.get('run_ptq_per_tensor', True):
        print("\n--- PTQ Per-Tensor Experiment (Model Generation) ---")
        try:
            # PTQ happens on eval_device (CPU)
            model_q = apply_ptq_per_tensor(fp32_model_state_dict_for_quant, num_classes, calib_loader, eval_device, config)
            path = os.path.join(save_dir, 'ptq_per_tensor_model.pth'); torch.jit.save(torch.jit.script(model_q.cpu()), path) # Ensure saving CPU model
            print(f"  PTQ Per-Tensor model saved to {path}")
            record_quant_results("ptq_per_tensor", "ptq_per_tensor", model_q, path)
        except Exception as e: print(f"  PTQ Per-Tensor failed: {e}"); traceback.print_exc()

    if config.get('run_ptq_per_channel_manual', True):
        print("\n--- PTQ Per-Channel (Manual QConfig) Experiment (Model Generation) ---")
        try:
            # PTQ happens on eval_device (CPU)
            model_q = apply_ptq_per_channel_manual(fp32_model_state_dict_for_quant, num_classes, calib_loader, eval_device, config)
            path = os.path.join(save_dir, 'ptq_per_channel_manual_model.pth'); torch.jit.save(torch.jit.script(model_q.cpu()), path) # Ensure saving CPU model
            print(f"  PTQ Per-Channel (Manual) model saved to {path}")
            record_quant_results("ptq_per_channel_manual", "ptq_per_channel_manual_qconfig", model_q, path)
        except Exception as e: print(f"  PTQ Per-Channel (Manual) failed: {e}"); traceback.print_exc()

    if config.get('run_qat', True):
        print("\n--- QAT Experiment (Model Generation) ---")
        try:
            # QAT training on training_device (GPU if available), then converted to CPU for final quantized model
            model_q, hist, train_time = apply_qat(fp32_model_state_dict_for_quant, num_classes, qat_train_loader, quick_val_loader, training_device, config)
            # model_q from apply_qat is already on CPU after conversion
            path = os.path.join(save_dir, 'qat_model.pth'); torch.jit.save(torch.jit.script(model_q.cpu()), path) # Ensure saving CPU model
            print(f"  QAT model saved to {path}. Training time: {train_time:.2f}s")
            qat_training_info = {"epochs": config['qat_epochs'], "learning_rate": config['qat_learning_rate'],
                "momentum": config['qat_momentum'], "weight_decay": config['qat_weight_decay'],
                "total_time_seconds": train_time, "history": dict(hist), "qat_backend": config.get('qat_backend', 'fbgemm')}
            record_quant_results("qat", "qat_int8", model_q, path, qat_training_info)
        except Exception as e: print(f"  QAT failed: {e}"); traceback.print_exc()

    if config.get('run_kmeans', False):
        print("\n--- K-means Quantization Experiment (Model Generation) ---")
        try:
            # K-means applied to baseline_fp32_model_cpu (already on CPU)
            model_q, cbooks = apply_kmeans_quantization(baseline_fp32_model_cpu, config.get('kmeans_clusters', 256))
            path = os.path.join(save_dir, 'kmeans_quantized_model.pth'); torch.save(model_q.state_dict(), path) # Saves state_dict
            print(f"  K-means model saved to {path}")
            kmeans_training_info = {"codebook_info_summary": {name: info['num_clusters'] for name, info in cbooks.items()}}
            # K-Means model is FP32 with clustered weights, evaluated on CPU (eval_device)
            record_quant_results("kmeans", f"kmeans_{config.get('kmeans_clusters', 256)}_clusters", model_q, path, kmeans_training_info)
        except Exception as e: print(f"  K-means quantization failed: {e}"); traceback.print_exc()

    results_path = os.path.join(save_dir, config['log_file_name'])
    with open(results_path, 'w') as f:
        def json_default_serializer(o):
            if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, defaultdict): return dict(o)
            if isinstance(o, torch.device): return str(o)
            try: return json.JSONEncoder.default(None, o)
            except TypeError: return str(o)
        json.dump(results, f, indent=2, default=json_default_serializer)
    print(f"\n--- Model Generation Complete --- \nResults saved to {results_path}")
    print("Note: Accuracy evaluations used quick_val_loader. Inference speed used benchmark_inference_loader_bs1.")
    return results

def setup_dummy_imagenet_mini(data_dir_root, num_classes=2, images_per_class_split=5):
    if not PIL_AVAILABLE: print("PIL/Pillow not available. Cannot create dummy images."); return
    if not os.path.exists(data_dir_root):
        print(f"Creating dummy data directory structure at: {data_dir_root}")
        os.makedirs(data_dir_root, exist_ok=True)
        for split in ['train', 'val']:
            split_path = os.path.join(data_dir_root, split)
            os.makedirs(split_path, exist_ok=True)
            for i in range(num_classes):
                class_name = f"class_{chr(ord('a') + i)}"; class_path = os.path.join(split_path, class_name)
                os.makedirs(class_path, exist_ok=True)
                for j in range(images_per_class_split):
                    try:
                        img = Image.new('RGB', (np.random.randint(224, 257), np.random.randint(224,257)), # Ensure images are at least 224x224
                                        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))
                        img.save(os.path.join(class_path, f"dummy_img_{split}_{i}_{j}.jpg"))
                    except Exception as e: print(f"Warning: Could not create dummy image: {e}")
        print(f"Dummy data created in {data_dir_root}.")

def setup_dummy_baseline_model(model_path, data_dir_for_num_classes, force_create=False):
    if not os.path.exists(model_path) or force_create:
        print(f"Creating dummy baseline model at: {model_path}")
        num_dummy_classes = 2; train_path = os.path.join(data_dir_for_num_classes, 'train')
        if os.path.exists(train_path) and os.path.isdir(train_path):
            subdirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
            if subdirs: num_dummy_classes = len(subdirs)
        print(f"Dummy model will have {num_dummy_classes} output classes.")
        dummy_model = torchvision.models.resnet50(weights=None) 
        dummy_model.fc = nn.Linear(dummy_model.fc.in_features, num_dummy_classes)
        torch.save(dummy_model.state_dict(), model_path)
        print(f"Dummy model saved to {model_path}.")

if __name__ == "__main__":
    config = {
        "data_dir": "./imagenet-mini",             # Your actual dataset
        "save_dir": "resnet50_quant_kmeans", # New save dir for this run
        "baseline_model_path": "./best_model.pth", # Your actual baseline
        "batch_size": 32,                         # Matched with pruning FT
        "num_workers": 4,                         # Matched with pruning FT
        "use_cuda_if_available": True,

        "ptq_calib_dataset_size": 200,
        "ptq_calib_batches": 10,

        "quick_check_val_samples": 200, # For accuracy reporting

        "inference_benchmark_samples": 1000, # For speed reporting
        "inference_benchmark_warmup_iters": 20,

        "run_ptq_per_tensor": False,
        "run_ptq_per_channel_manual": False,
        "run_qat": False,
        "run_kmeans": True, # Or True if you want to include it

        "ptq_backend": "fbgemm",
        "qat_backend": "fbgemm",
        "qat_epochs": 8,                          # <<< SET TO 8
        "qat_learning_rate": 1e-5,
        "qat_momentum": 0.9,
        "qat_weight_decay": 0.0001,
        "qat_log_interval": 10, # How often to print QAT training loss

        "kmeans_clusters": 256,
        "log_file_name": "quant_methods_comparison_results.json" # Final log file
    }
    
    # Comment these out if you are using your real data/model and they already exist
    # Ensure enough dummy images if benchmark_samples is high (e.g., 1000)
    # For 2 classes, 1000 samples for val -> images_per_class_split = 500 for val
    # Make dummy data larger if needed for benchmark_inference_loader_bs1
    num_dummy_classes_main = 2
    images_per_class_main = max(5, config["inference_benchmark_samples"] // num_dummy_classes_main + 10) # Ensure enough val images
    if not os.path.exists(config['data_dir']) or not os.listdir(os.path.join(config['data_dir'], 'val')):
         setup_dummy_imagenet_mini(config['data_dir'], num_classes=num_dummy_classes_main, images_per_class_split=images_per_class_main)
    else:
        print(f"Dataset at {config['data_dir']} seems to exist. Skipping dummy data creation. Ensure it has enough validation samples for inference benchmark.")

    if not os.path.exists(config['baseline_model_path']):
        setup_dummy_baseline_model(config['baseline_model_path'], config['data_dir'])
    else:
        print(f"Baseline model at {config['baseline_model_path']} exists. Skipping dummy model creation.")
        
    results = run_quantization_experiments(config)
    if results:
        print("\nModel generation script (with benchmark-style inference) finished successfully.")
    else:
        print("\nModel generation script (with benchmark-style inference) failed or was aborted.")