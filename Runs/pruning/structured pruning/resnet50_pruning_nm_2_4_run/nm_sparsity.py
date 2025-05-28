import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# from torchvision.models import resnet50, ResNet50_Weights # Now using torchvision.models directly
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import os
import json
import time
import copy

# import argparse # Not used if CONFIG is hardcoded

# --- Configuration ---
# Edit these parameters to configure your N:M sparsity experiment
CONFIG = {
    # --- General Paths and Data ---
    "data_dir": "./imagenet-mini",  # Path to your dataset
    "save_dir": "resnet50_pruning_nm_2_4_run", # Directory to save models and logs
    "baseline_model_path": "./best_model.pth", # Path to your trained baseline model

    # --- Model and Training (Primarily for baseline, if you were to train it here) ---
    "num_epochs_baseline_train": 1, # Number of epochs if training baseline from scratch (not used if loading)
    "batch_size": 32,
    "learning_rate_baseline_train": 0.001, # LR if training baseline from scratch
    "num_workers": 4,
    "use_pretrained_imagenet_weights": False, # If True, loads ImageNet weights, adapts FC, ignores baseline_model_path for loading

    # --- Pruning Technique Configuration ---
    "pruning_technique": "nm_sparsity", # Indicates the type of pruning

    # --- N:M Sparsity Specific Parameters ---
    "nm_sparsity_n": 2,                 # N: The number of weights to keep in a block
    "nm_sparsity_m": 4,                 # M: The size of the block
    "nm_ignored_module_names": ["fc"],  # List of module names (e.g., 'fc', 'conv1') to EXCLUDE from N:M pruning

    # --- Fine-Tuning Configuration ---
    "ft_epochs": 15,                    # Number of epochs for fine-tuning after pruning
    "ft_learning_rate": 5e-5,           # Learning rate for fine-tuning
    "ft_momentum": 0.9,
    "ft_weight_decay": 1e-4,

    # --- Experiment Control ---
    "evaluate_only": False,             # If True, loads 'pruned_model_to_evaluate' and only runs evaluation
    "pruned_model_to_evaluate": "",     # Path to a pre-pruned model if evaluate_only is True

    # --- Logging ---
    "log_file_name": "nm_sparsity_results_v2.json", # Name of the JSON log file

    # --- Optional: For model definition if not auto-detected ---
    # "num_classes": 100 # Number of classes in your dataset, if not reliably auto-detected from data_dir
}

# --- Helper Functions (get_device, get_data_loaders, count_parameters, etc.) ---
# ... (all your helper functions like get_device, get_data_loaders, count_parameters, 
#      get_model_size_mb, train_one_epoch, evaluate_model, fine_tune_model, 
#      apply_nm_sparsity_pruning, calculate_achieved_sparsity_percent should be here)
# --- Helper Functions ---

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_classes_from_data_dir(data_dir):
    try:
        train_path = os.path.join(data_dir, 'train')
        if os.path.exists(train_path) and os.path.isdir(train_path):
            class_names = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
            num_classes = len(class_names)
            if num_classes > 0:
                print(f"Auto-detected {num_classes} classes from {train_path}")
                return num_classes
        print(f"Warning: Could not auto-detect num_classes from {train_path}.")
        return None 
    except Exception as e:
        print(f"Error auto-detecting num_classes: {e}")
        return None

def get_data_loaders(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes detected from dataset: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader_inf = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader_inf, num_classes

def get_total_model_parameters(model):
    """Counts all elements in nn.Parameter objects."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_effective_non_zero_parameters(model):
    """
    Counts non-zero elements in the *effective* weights (respecting pruning masks)
    and other parameters like biases.
    """
    non_zero_params = 0
    # Use model.named_modules() to easily access .weight and .bias attributes
    # which resolve to effective weights if pruning is active.
    for name, module in model.named_modules():
        # Check if this module is a "leaf" module that actually holds weights/biases
        # to avoid double counting if parent modules somehow also have these attributes.
        # For nn.Linear and nn.Conv2d, this is generally safe.
        if isinstance(module, (nn.Linear, nn.Conv2d)): # Be more specific
            if hasattr(module, 'weight') and module.weight is not None:
                # module.weight IS the effective (masked) weight if hooks are active
                non_zero_params += torch.count_nonzero(module.weight.data).item()
            
            if hasattr(module, 'bias') and module.bias is not None:
                non_zero_params += torch.count_nonzero(module.bias.data).item()
        # Add other module types if they have parameters you care about (e.g., BatchNorm)
        # Note: BatchNorm parameters are usually not pruned with N:M weight pruning
        elif isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None: # This is gamma
                non_zero_params += torch.count_nonzero(module.weight.data).item()
            if hasattr(module, 'bias') and module.bias is not None: # This is beta
                non_zero_params += torch.count_nonzero(module.bias.data).item()

    return non_zero_params

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch_num, total_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0: 
            print(f"Epoch [{epoch_num+1}/{total_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model(model, criterion, data_loader, device, measure_speed=False, test_loader_inf=None, num_inf_batches=1000):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    inference_metrics = {}

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_loss = running_loss / total_samples if total_samples > 0 else float('inf')
    val_acc = correct_predictions / total_samples if total_samples > 0 else 0.0

    if measure_speed and test_loader_inf:
        print(f"Measuring inference speed for {num_inf_batches} images (batch_size=1)...")
        warmup_iterations = 20
        for _ in range(warmup_iterations): 
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            _ = model(dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()
        
        total_time = 0
        images_processed = 0
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader_inf):
                if images_processed >= num_inf_batches: break
                inputs = inputs.to(device)
                start_time = time.perf_counter()
                _ = model(inputs)
                if device.type == 'cuda': torch.cuda.synchronize()
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
                images_processed += inputs.size(0)
        
        images_per_second = images_processed / total_time if total_time > 0 else 0
        latency_ms_per_image = (total_time / images_processed * 1000) if images_processed > 0 else 0
        
        inference_metrics = {
            "images_per_second": images_per_second,
            "latency_ms_per_image": latency_ms_per_image,
            "total_images_measured": images_processed,
            "total_time_seconds": total_time
        }
        print(f"Inference speed: {images_per_second:.2f} img/s, Latency: {latency_ms_per_image:.2f} ms/img")

    return val_loss, val_acc, inference_metrics

def fine_tune_model(model, criterion, train_loader, val_loader, test_loader_inf, device, config, sparsity_info_str):
    print(f"Starting fine-tuning for {config['ft_epochs']} epochs ({sparsity_info_str})...")
    optimizer = optim.SGD(model.parameters(), lr=config['ft_learning_rate'], momentum=config['ft_momentum'], weight_decay=config['ft_weight_decay'])
    
    best_val_acc = 0.0
    ft_history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    ft_start_time = time.time()

    for epoch in range(config['ft_epochs']):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, config['ft_epochs'])
        val_loss, val_acc, _ = evaluate_model(model, criterion, val_loader, device)
        
        print(f"FT Epoch [{epoch+1}/{config['ft_epochs']}] ({sparsity_info_str}) Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        ft_history["loss"].append(train_loss)
        ft_history["accuracy"].append(train_acc)
        ft_history["val_loss"].append(val_loss)
        ft_history["val_accuracy"].append(val_acc)
        
        if val_acc > best_val_acc: best_val_acc = val_acc

    ft_total_time_seconds = time.time() - ft_start_time
    print(f"Fine-tuning ({sparsity_info_str}) completed in {ft_total_time_seconds:.2f} seconds.")
    return ft_history, ft_total_time_seconds

def apply_nm_sparsity_pruning(model, N=2, M=4, ignored_module_names=[]):
    print(f"Applying {N}:{M} sparsity pruning...")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name not in ignored_module_names:
                weight = module.weight.data
                orig_shape = weight.shape
                # Flatten along the last dim into M-sized blocks
                reshaped = weight.view(-1, M)
                abs_weight = reshaped.abs()

                # Keep top-N values in each block
                topk = torch.topk(abs_weight, k=N, dim=1, largest=True, sorted=False)
                mask = torch.zeros_like(reshaped)
                mask.scatter_(1, topk.indices, 1.0)

                # Apply mask and reshape back
                pruned_weight = reshaped * mask
                module.weight.data = pruned_weight.view(orig_shape)
                print(f"Pruned {name} to {N}:{M} sparsity.")

    return model

def calculate_achieved_sparsity_percent(baseline_total_params, current_non_zero_params):
    if baseline_total_params == 0: return 0.0
    return (baseline_total_params - current_non_zero_params) / baseline_total_params * 100


class NMPruningMethod(prune.BasePruningMethod):
    """
    Prunes weights to achieve N:M block sparsity.
    In each block of M consecutive weights (flattened), only N weights with 
    the largest absolute values are kept.
    """
    PRUNING_TYPE = 'unstructured' # This is important. Even though N:M is "structured"
                                 # in blocks, from the perspective of individual weight
                                 # connections, it's 'unstructured' as any N can be chosen.
                                 # 'structured' pruning in PyTorch usually implies pruning
                                 # entire channels/filters.

    def __init__(self, N, M):
        super(NMPruningMethod, self).__init__()
        if not isinstance(N, int) or not isinstance(M, int) or N <= 0 or M <= 0 or N > M:
            raise ValueError(f"N ({N}) and M ({M}) must be positive integers with N <= M.")
        self.N = N
        self.M = M

    def compute_mask(self, t, default_mask):
        """
        Computes the N:M mask for a given tensor 't'.
        't' is the parameter tensor to be pruned (e.g., module.weight).
        'default_mask' is the existing mask of 't' (usually a mask of all ones
                       if no prior pruning was applied).
        """
        # Clone to avoid modifying the original tensor data if it's passed by 't'
        # In practice, 't' is often a detached copy or the buffer itself.
        weight_data = t.clone().detach() 
        orig_shape = weight_data.shape
        numel = weight_data.numel()

        if numel == 0: # Handle empty tensors
            return default_mask 

        # If the tensor isn't perfectly divisible by M, we have a few choices:
        # 1. Error: Simplest, forces layers to be compatible.
        # 2. Pad conceptually (complex to implement correctly within compute_mask alone for all shapes).
        # 3. Prune only the part that is divisible, leave the rest unpruned (safest for a general method).
        # Here, we'll try to prune the largest divisible part and keep the remainder.
        
        num_blocks = numel // self.M
        
        if num_blocks == 0: # Tensor is smaller than M, cannot form a single N:M block
            # print(f"  Warning: Tensor with {numel} elements is smaller than M={self.M}. Not applying N:M mask.")
            return default_mask # Keep all weights (or whatever default_mask dictates)

        # Reshape only the part that forms complete M-blocks
        reshaped_part = weight_data.flatten()[:num_blocks * self.M].view(-1, self.M)
        
        # Compute N:M mask for the reshaped part
        abs_weight = reshaped_part.abs()
        topk = torch.topk(abs_weight, k=self.N, dim=1, largest=True, sorted=False)
        
        mask_for_reshaped_part = torch.zeros_like(reshaped_part, dtype=torch.bool) # Mask should be bool
        mask_for_reshaped_part.scatter_(1, topk.indices, True)
        
        # Flatten this mask
        final_mask_flat = mask_for_reshaped_part.flatten()

        # If there's a remainder, append a mask of ones (keep unpruned)
        if numel % self.M != 0:
            remainder_size = numel - (num_blocks * self.M)
            if remainder_size > 0:
                remainder_mask = torch.ones(remainder_size, dtype=torch.bool, device=t.device)
                final_mask_flat = torch.cat((final_mask_flat, remainder_mask))
        
        # Reshape the complete flat mask back to the original tensor shape
        final_mask = final_mask_flat.view(orig_shape)
        
        # The pruning framework expects the new mask to be combined with the default_mask
        # (e.g., if some parts were already pruned by a previous step).
        return final_mask # Return the new mask, let prune utility multiply with default_mask

    @classmethod
    def apply(cls, module, name, N, M, **kwargs): # Add N, M to the apply signature
        """
        Adds the N:M pruning hook to the module.
        'module': the module to prune.
        'name': the name of the parameter to prune (e.g., 'weight', 'bias').
        'N', 'M': N:M sparsity parameters.
        """
        return super(NMPruningMethod, cls).apply(module, name, N=N, M=M, **kwargs) # Pass N, M to __init__

# Helper function to apply this custom pruning
def apply_nm_sparsity_with_pruning_utils(model, N, M, ignored_module_names=[]):
    print(f"Applying {N}:{M} sparsity using torch.nn.utils.prune...")
    print(f"Ignoring modules: {ignored_module_names}")
    applied_count = 0
    skipped_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if name in ignored_module_names:
                print(f"  Skipping {name}: in ignored_module_names.")
                continue
            
            if not hasattr(module, 'weight') or module.weight is None:
                print(f"  Skipping {name}: no 'weight' or weight is None.")
                continue

            # Check for numel divisibility for cleaner application, though the mask handles it.
            # It's good practice to be aware of which layers might have remainders.
            if module.weight.numel() % M != 0:
                 print(f"  Note: {name} (numel={module.weight.numel()}) is not perfectly divisible by M={M}. "
                       f"The N:M mask will apply to full blocks, remainder will be unpruned.")
            
            if module.weight.numel() < M :
                print(f"  Skipping {name}: (numel={module.weight.numel()}) is smaller than M={M}. Cannot apply N:M.")
                skipped_count += 1
                continue

            NMPruningMethod.apply(module, name='weight', N=N, M=M) # Use the classmethod
            print(f"  Applied N:M pruning hook to {name}.")
            applied_count += 1
        
    print(f"\nN:M Pruning Hooks Applied: {applied_count} layers. Skipped (too small): {skipped_count} layers.")
    return model

# Helper function to make pruning permanent
def make_pruning_permanent(model):
    print("Making pruning permanent by removing hooks...")
    for name, module in model.named_modules():
        if prune.is_pruned(module): # Check if any pruning hook (like 'weight_mask') exists
            # Iterate through named buffers to find masks, as multiple params could be pruned.
            # A bit more robust than assuming only 'weight' is pruned.
            # However, for this specific N:M, we know it's 'weight'.
            if hasattr(module, 'weight_mask'): # Check if 'weight' was pruned
                prune.remove(module, 'weight')
                print(f"  Made pruning permanent for 'weight' in {name}.")
            # Add similar checks if you prune other parameters like 'bias'
    return model


# --- Main Script ---
def main(config):
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader_inf, num_classes_from_data = get_data_loaders(
        config['data_dir'], config['batch_size'], config['num_workers']
    )
    
    num_classes = num_classes_from_data
    if num_classes is None:
        num_classes_config = config.get("num_classes") 
        if num_classes_config:
            num_classes = num_classes_config
            print(f"Using num_classes from CONFIG: {num_classes}")
        else:
            raise ValueError("Number of classes could not be determined. Please check data_dir or set 'num_classes' in CONFIG.")
    
    criterion = nn.CrossEntropyLoss()
    all_results = {
        "experiment_type": f"{config['pruning_technique']}_pruning",
        "config": config,
        "baseline_results": {},
        "pruning_runs": [] 
    }

    # --- Baseline Model ---
    print("Loading/Setting up baseline model...")
    baseline_model = torchvision.models.resnet50(weights=None, num_classes=num_classes) 
    
    baseline_model_path_str = "Custom Trained or Adapted" 
    if config["use_pretrained_imagenet_weights"]:
        print("Loading ImageNet pretrained weights for ResNet50 and adapting FC layer...")
        pt_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = pt_model.fc.in_features
        pt_model.fc = nn.Linear(num_ftrs, num_classes)
        baseline_model = pt_model 
        baseline_model_path_str = "ImageNet Pretrained (adapted)"
    elif os.path.exists(config['baseline_model_path']):
        print(f"Loading baseline model from: {config['baseline_model_path']}")
        # Add weights_only=True for security and future compatibility
        baseline_model.load_state_dict(torch.load(config['baseline_model_path'], map_location=device, weights_only=True))
        baseline_model_path_str = config['baseline_model_path']
    else:
        raise FileNotFoundError(
            f"Baseline model not found at {config['baseline_model_path']} and "
            f"'use_pretrained_imagenet_weights' is False. A baseline model is required."
        )

    baseline_model.to(device)
    print("Evaluating baseline model...")
    baseline_val_loss, baseline_val_acc, baseline_inf_metrics = evaluate_model(
        baseline_model, criterion, val_loader, device, measure_speed=True, test_loader_inf=test_loader_inf
    )
    
    # Use the new parameter counting functions for baseline
    baseline_total_params_val = get_total_model_parameters(baseline_model)
    baseline_non_zero_params_val = count_effective_non_zero_parameters(baseline_model) # For baseline, effective non-zero IS total non-zero
    baseline_parameter_counts = {
        "total_params": baseline_total_params_val,
        "non_zero_params": baseline_non_zero_params_val
    }
    baseline_size_mb = get_model_size_mb(baseline_model)

    all_results["baseline_results"] = {
        "run_type": "baseline_evaluation",
        "model_path": baseline_model_path_str,
        "val_accuracy": baseline_val_acc,
        "val_loss": baseline_val_loss,
        "model_size_mb": baseline_size_mb, 
        "parameter_counts": baseline_parameter_counts, # Store the dict
        "inference_metrics": baseline_inf_metrics
    }
    print(f"Baseline Results: Acc: {baseline_val_acc:.4f}, Loss: {baseline_val_loss:.4f}, Size (dense): {baseline_size_mb:.2f}MB")
    print(f"  Params: Total={baseline_parameter_counts['total_params']}, Non-Zero={baseline_parameter_counts['non_zero_params']}")
    if baseline_inf_metrics:
        print(f"  Baseline Inference: {baseline_inf_metrics.get('images_per_second', 0):.2f} IPS, {baseline_inf_metrics.get('latency_ms_per_image',0):.2f} ms/img")

    # --- Evaluate Only Mode ---
    if config['evaluate_only']:
        if not config['pruned_model_to_evaluate'] or not os.path.exists(config['pruned_model_to_evaluate']):
            raise FileNotFoundError(
                f"Evaluate_only is True, but 'pruned_model_to_evaluate' path is invalid: "
                f"{config['pruned_model_to_evaluate']}"
            )
        
        print(f"\n--- Evaluating Pre-Pruned Model: {config['pruned_model_to_evaluate']} ---")
        eval_model = torchvision.models.resnet50(weights=None, num_classes=num_classes) 
        # Add weights_only=True
        eval_model.load_state_dict(torch.load(config['pruned_model_to_evaluate'], map_location=device, weights_only=True))
        eval_model.to(device)

        eval_val_loss, eval_val_acc, eval_inf_metrics = evaluate_model(
            eval_model, criterion, val_loader, device, measure_speed=True, test_loader_inf=test_loader_inf
        )
        # For a saved model, pruning is permanent, so effective non-zero is the actual non-zero.
        eval_total_params_val = get_total_model_parameters(eval_model)
        eval_non_zero_params_val = count_effective_non_zero_parameters(eval_model)
        eval_parameter_counts = {
            "total_params": eval_total_params_val,
            "non_zero_params": eval_non_zero_params_val
        }
        eval_size_mb = get_model_size_mb(eval_model)
        # Use the baseline_total_params_val from the earlier calculation for the denominator
        eval_achieved_sparsity = calculate_achieved_sparsity_percent(
            baseline_total_params_val, eval_non_zero_params_val 
        )

        print(f"Evaluation Only Results for {config['pruned_model_to_evaluate']}:")
        print(f"  Acc: {eval_val_acc:.4f}, Loss: {eval_val_loss:.4f}, Size (dense): {eval_size_mb:.2f}MB")
        print(f"  Params: Total={eval_parameter_counts['total_params']}, Non-Zero={eval_parameter_counts['non_zero_params']}")
        print(f"  Achieved Overall Sparsity (vs baseline total): {eval_achieved_sparsity:.2f}%")
        if eval_inf_metrics:
            print(f"  Inference: {eval_inf_metrics.get('images_per_second',0):.2f} IPS, {eval_inf_metrics.get('latency_ms_per_image',0):.2f} ms/img")
        
        all_results["evaluation_only_run"] = {
            "model_path": config['pruned_model_to_evaluate'],
            "val_accuracy": eval_val_acc,
            "val_loss": eval_val_loss,
            "model_size_mb": eval_size_mb,
            "parameter_counts": eval_parameter_counts,
            "achieved_overall_sparsity_percent": eval_achieved_sparsity,
            "inference_metrics": eval_inf_metrics
        }
        log_path = os.path.join(config['save_dir'], config['log_file_name'])
        with open(log_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Evaluation only results saved to {log_path}")
        return

    # --- Pruning and Fine-tuning (if not evaluate_only) ---
    if config["pruning_technique"] == "nm_sparsity":
        n, m = config['nm_sparsity_n'], config['nm_sparsity_m']
        sparsity_info_str = f"{n}:{m} Sparsity"
        print(f"\n--- Processing {sparsity_info_str} ---")
        
        current_model = copy.deepcopy(baseline_model)
        current_model.to(device)

        # 1. Apply N:M Sparsity Pruning (hooks)
        current_model = apply_nm_sparsity_with_pruning_utils(
            current_model, 
            N=n, 
            M=m, 
            ignored_module_names=config['nm_ignored_module_names']
        )
        
        print("\nEvaluating model immediately after N:M pruning hooks (before fine-tuning)...")
        pruned_eval_loss, pruned_eval_acc, _ = evaluate_model(current_model, criterion, val_loader, device)
        
        # Parameter counting when hooks are active
        # Total params counts 'weight_orig' if it's a parameter, which is correct for storage.
        # Effective non-zero counts non-zeros in the masked 'module.weight'.
        params_after_hooks_total_val = get_total_model_parameters(current_model)
        params_after_hooks_non_zero_val = count_effective_non_zero_parameters(current_model)
        params_after_pruning_hooks_counts = {
            "total_params": params_after_hooks_total_val,
            "non_zero_params": params_after_hooks_non_zero_val
        }
        
        size_mb_after_pruning_hooks = get_model_size_mb(current_model) # Will be larger due to mask buffers
        
        # Sparsity is (baseline_total - current_effective_non_zero) / baseline_total
        achieved_sparsity_before_ft = calculate_achieved_sparsity_percent(
            baseline_total_params_val, # Denominator is always the original baseline total
            params_after_hooks_non_zero_val
        )
        
        print(f"After N:M Pruning Hooks ({sparsity_info_str}): Acc: {pruned_eval_acc:.4f}, Loss: {pruned_eval_loss:.4f}")
        print(f"  Size (with hooks): {size_mb_after_pruning_hooks:.2f}MB, Params: Total(storage)={params_after_hooks_total_val}, Effective Non-Zero={params_after_hooks_non_zero_val}")
        print(f"  Achieved Overall Sparsity (vs baseline total): {achieved_sparsity_before_ft:.2f}%")

        # 2. Fine-tune the N:M pruned model (hooks are active)
        ft_history, ft_time = fine_tune_model(
            current_model, criterion, train_loader, val_loader, test_loader_inf, device, config, sparsity_info_str
        )

        # 3. Make Pruning Permanent after Fine-Tuning
        current_model = make_pruning_permanent(current_model)
        # Now, 'weight_mask' and 'weight_orig' are gone. 'module.weight' directly contains the zeroed values.
        
        print("\nEvaluating fine-tuned N:M pruned model (pruning made permanent)...")
        ft_val_loss, ft_val_acc, ft_inf_metrics = evaluate_model(
            current_model, criterion, val_loader, device, measure_speed=True, test_loader_inf=test_loader_inf
        )
        
        # Parameter counting after making pruning permanent
        ft_total_params_val = get_total_model_parameters(current_model) # Should now be similar to baseline_total
        ft_non_zero_params_val = count_effective_non_zero_parameters(current_model) # Reflects final sparsity
        ft_parameter_counts = {
            "total_params": ft_total_params_val,
            "non_zero_params": ft_non_zero_params_val
        }
        
        ft_size_mb = get_model_size_mb(current_model) # Should be similar to baseline model size
        
        achieved_sparsity_final = calculate_achieved_sparsity_percent(
            baseline_total_params_val, # Denominator is always the original baseline total
            ft_non_zero_params_val
        )

        pruned_model_filename = f"resnet50_nm_{n}_{m}_ft.pth"
        pruned_model_save_path = os.path.join(config['save_dir'], pruned_model_filename)
        torch.save(current_model.state_dict(), pruned_model_save_path)

        run_summary = {
            "run_type": f"pruning_run_nm_sparsity",
            "pruning_method_name": f"{n}:{m}_semi_structured_sparsity",
            "nm_config": {"N": n, "M": m, "ignored_modules": config['nm_ignored_module_names']},
            "achieved_overall_parameter_sparsity_percent": achieved_sparsity_final,
            "pruned_model_saved_as": pruned_model_save_path,
            "evaluation_after_pruning_before_ft": { # This now reflects state with hooks
                "val_accuracy": pruned_eval_acc,
                "val_loss": pruned_eval_loss,
                "model_size_mb": size_mb_after_pruning_hooks, 
                "parameter_counts": params_after_pruning_hooks_counts, 
            },
            "fine_tuning_config": {
                "epochs": config['ft_epochs'],
                "learning_rate": config['ft_learning_rate'],
                "momentum": config['ft_momentum'],
                "weight_decay": config['ft_weight_decay'],
                "total_time_seconds": ft_time,
                "history": ft_history
            },
            "final_evaluation_metrics": {
                "val_accuracy": ft_val_acc,
                "val_loss": ft_val_loss,
                "model_size_mb": ft_size_mb, 
                "parameter_counts": ft_parameter_counts, 
                "inference_metrics": ft_inf_metrics
            }
        }
        all_results["pruning_runs"].append(run_summary)

        print(f"Fine-tuned N:M Pruned Model ({sparsity_info_str}): Acc: {ft_val_acc:.4f}, Loss: {ft_val_loss:.4f}")
        print(f"  Size (dense): {ft_size_mb:.2f}MB, Params: Total={ft_parameter_counts['total_params']}, Non-Zero={ft_parameter_counts['non_zero_params']}")
        print(f"  Achieved Overall Sparsity (vs baseline total): {achieved_sparsity_final:.2f}%")
        if ft_inf_metrics:
            print(f"  Inference: {ft_inf_metrics.get('images_per_second',0):.2f} IPS, {ft_inf_metrics.get('latency_ms_per_image',0):.2f} ms/img")

    else:
        print(f"Unsupported pruning_technique in CONFIG: {config['pruning_technique']}")

    log_path = os.path.join(config['save_dir'], config['log_file_name'])
    with open(log_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n--- Experiment Complete ---")
    print(f"All results saved to {log_path}")


if __name__ == '__main__':
    # Parameters are set by editing the CONFIG dictionary at the top of the script.

    # --- Configuration Checks and Dynamic Updates (based on CONFIG) ---
    
    # Dynamically update save_dir and log_file_name if using non-default N:M values
    # and the config still has the default names. This helps organize multiple N:M experiments.
    default_nm_save_dir = "resnet50_pruning_nm_2_4_run_v2" # Match default in CONFIG
    default_nm_log_file = "nm_sparsity_results_v2.json"   # Match default in CONFIG

    # Check if N or M are different from the ones used to generate the default names
    is_default_nm_config = (CONFIG['nm_sparsity_n'] == 2 and CONFIG['nm_sparsity_m'] == 4)

    if not is_default_nm_config:
        nm_str = f"nm_{CONFIG['nm_sparsity_n']}_{CONFIG['nm_sparsity_m']}"
        # Only update if the current save_dir/log_file is the default one for 2:4
        if CONFIG['save_dir'] == default_nm_save_dir:
            CONFIG['save_dir'] = f"resnet50_pruning_{nm_str}_run"
        if CONFIG['log_file_name'] == default_nm_log_file:
            CONFIG['log_file_name'] = f"{nm_str}_pruning_results.json"
    
    # --- Pre-run Sanity Checks (using values from CONFIG) ---
    if not os.path.exists(CONFIG['data_dir']):
       print(f"Error: data_dir '{CONFIG['data_dir']}' does not exist. Please check CONFIG.")
       exit(1)
       
    if not CONFIG['evaluate_only'] and \
       not CONFIG['use_pretrained_imagenet_weights'] and \
       not os.path.exists(CONFIG['baseline_model_path']):
       print(f"Error: baseline_model_path '{CONFIG['baseline_model_path']}' does not exist, "
             "and 'use_pretrained_imagenet_weights' is False. "
             "A baseline model is required for pruning/fine-tuning. Check CONFIG.")
       exit(1)

    if CONFIG['evaluate_only'] and (not CONFIG['pruned_model_to_evaluate'] or not os.path.exists(CONFIG['pruned_model_to_evaluate'])):
        print(f"Error: 'evaluate_only' is True, but 'pruned_model_to_evaluate' path "
              f"'{CONFIG['pruned_model_to_evaluate']}' is invalid or does not exist. Check CONFIG.")
        exit(1)

    # --- Run Main Experiment Function ---
    main(CONFIG)