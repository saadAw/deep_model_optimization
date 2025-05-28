import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import copy
import time
import json
import datetime
from tqdm import tqdm

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = './imagenet-mini'
TEACHER_MODEL_PATH = './models/best_model.pth' # Your fine-tuned ResNet50
STUDENT_MODEL_SAVE_PATH = './models/student_resnet18_distilled_pretrained.pth'
RESULTS_JSON_PATH = './distillation_results_pretrained.json'
STUDENT_ARCHITECTURE = 'resnet18'

NUM_CLASSES = -1 # Auto-detected or set manually, e.g., 100

# Comparable settings from your pruning script
BATCH_SIZE = 32  # Match pruning script's batch_size for val_loader
EPOCHS = 15      # To match ft_epochs for time comparison of optimization phase
FT_LEARNING_RATE_EQUIVALENT = 5e-5 # Student LR, can be different, but good to note
FT_OPTIMIZER_TYPE = 'Adam' # For student, pruning script used SGD for FT

# Distillation specific
STUDENT_PRETRAINED = True # NEW: To easily toggle
INITIAL_STUDENT_LR = 1e-4   # ADJUSTED: Smaller LR for fine-tuning a pretrained student
TEMPERATURE = 4.0
ALPHA = 0.7

# Scheduler Config (for StepLR example)
SCHEDULER_STEP_SIZE = 5
SCHEDULER_GAMMA = 0.1 # Reduce LR by this factor

NUM_WORKERS = 4
PIN_MEMORY = True
# For inference speed measurement, from your pruning script
NUM_INF_BATCHES_SPEED_TEST = 10

# --- 1. Data Loading and Preprocessing (Aligned with Pruning Script) ---
def get_data_loaders(data_dir, batch_size, num_workers, pin_memory, num_classes_from_data=False):
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
    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

        if not train_dataset.samples or not val_dataset.samples:
            raise FileNotFoundError("Dataset not found or empty.")

        print(f"Found {len(train_dataset)} training images belonging to {len(train_dataset.classes)} classes.")
        print(f"Found {len(val_dataset)} validation images belonging to {len(val_dataset.classes)} classes.")
        actual_num_classes = len(train_dataset.classes)

        if num_classes_from_data:
            global NUM_CLASSES
            NUM_CLASSES = actual_num_classes
            print(f"Number of classes set from data: {NUM_CLASSES}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        # val_loader batch_size from global BATCH_SIZE (which is 32 here)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        # test_loader_inf for inference speed measurement, batch_size=1
        test_loader_inf = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader_inf, actual_num_classes
    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, 0

# --- 2. Model Definition and Loading (Same as before) ---
def get_model(model_name, num_classes, pretrained=True):
    model_instance = None
    if model_name == 'resnet18':
        model_instance = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model_instance.fc = nn.Linear(model_instance.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        model_instance = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        model_instance.fc = nn.Linear(model_instance.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model_instance = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model_instance.fc = nn.Linear(model_instance.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    return model_instance.to(DEVICE)

# --- 3. Loss Functions (Same as before) ---
criterion_hard = nn.CrossEntropyLoss().to(DEVICE) # Used by evaluate_model too
criterion_distill = nn.KLDivLoss(reduction='batchmean').to(DEVICE)

# --- 4. Metric Helper Functions (from Pruning Script) ---
def get_total_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_effective_non_zero_parameters(model): # For dense models, this is same as total
    non_zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                non_zero_params += torch.count_nonzero(module.weight.data).item()
            if hasattr(module, 'bias') and module.bias is not None:
                non_zero_params += torch.count_nonzero(module.bias.data).item()
        elif isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None:
                non_zero_params += torch.count_nonzero(module.weight.data).item()
            if hasattr(module, 'bias') and module.bias is not None:
                non_zero_params += torch.count_nonzero(module.bias.data).item()
    return non_zero_params

def get_model_size_mb(model): # From pruning script
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

# Evaluate model function from your pruning script
def evaluate_model_from_pruning_script(model, criterion, data_loader, device, measure_speed=False, test_loader_inf=None, num_inf_batches=10, eval_desc="Evaluating"):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    inference_metrics = {}

    progress_bar = tqdm(data_loader, desc=eval_desc, leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())


    val_loss = running_loss / total_samples if total_samples > 0 else float('inf')
    val_acc = correct_predictions / total_samples if total_samples > 0 else 0.0

    if measure_speed and test_loader_inf:
        print(f"Measuring inference speed for up to {num_inf_batches} images (batch_size=1)...")
        warmup_iterations = 5
        for _ in range(warmup_iterations):
            try:
                dummy_input, _ = next(iter(test_loader_inf))
                dummy_input = dummy_input.to(device)
                _ = model(dummy_input)
            except StopIteration: # Handle if test_loader_inf is smaller than warmup_iterations
                break
        if device.type == 'cuda': torch.cuda.synchronize()

        total_time = 0
        images_processed = 0
        with torch.no_grad():
            for i, (inputs_inf, _) in enumerate(test_loader_inf):
                if images_processed >= num_inf_batches: break
                inputs_inf = inputs_inf.to(device)
                start_time = time.perf_counter()
                _ = model(inputs_inf)
                if device.type == 'cuda': torch.cuda.synchronize()
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
                images_processed += inputs_inf.size(0)
                if (i+1) % 200 == 0: print(f"  Processed {images_processed}/{num_inf_batches} for speed test...")


        images_per_second = images_processed / total_time if total_time > 0 else 0
        latency_ms_per_image = (total_time / images_processed * 1000) if images_processed > 0 else 0

        inference_metrics = {
            "images_per_second": images_per_second,
            "latency_ms_per_image": latency_ms_per_image,
            "total_images_measured": images_processed,
            "total_time_seconds": total_time
        }
        print(f"Inference speed: {images_per_second:.2f} img/s, Latency: {latency_ms_per_image:.2f} ms/img over {images_processed} images.")

    return val_loss, val_acc, inference_metrics


# --- 5. Training and Validation Loops ---
def train_epoch_kd(student_model, teacher_model, loader, optimizer, temp, alpha, current_epoch, total_epochs): # Added epoch tracking for print
    student_model.train()
    teacher_model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    epoch_start_time = time.time()

    progress_bar = tqdm(loader, desc=f"Epoch {current_epoch+1}/{total_epochs} Training", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        student_outputs = student_model(inputs)
        loss_hard = criterion_hard(student_outputs, labels)
        soft_teacher_targets = F.softmax(teacher_outputs / temp, dim=1)
        soft_student_log_probs = F.log_softmax(student_outputs / temp, dim=1)
        loss_distill = criterion_distill(soft_student_log_probs, soft_teacher_targets) * (temp * temp)
        total_loss = (1.0 - alpha) * loss_hard + alpha * loss_distill
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item() * inputs.size(0)
        _, preds = torch.max(student_outputs, 1)
        correct_predictions += torch.sum(preds == labels.data).item() # .item()
        total_samples += inputs.size(0)
        if (i + 1) % 100 == 0:
             progress_bar.set_postfix_str(f"Loss: {total_loss.item():.4f}")


    epoch_loss = running_loss / total_samples if total_samples > 0 else float('inf')
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    epoch_duration = time.time() - epoch_start_time
    return epoch_loss, epoch_acc, epoch_duration

# Validation during training - uses the core of evaluate_model_from_pruning_script for consistency, but no speed test
def validate_epoch_kd(model, criterion, loader, device, current_epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    epoch_start_time = time.time()

    progress_bar = tqdm(loader, desc=f"Epoch {current_epoch+1}/{total_epochs} Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item() # .item()
            progress_bar.set_postfix(loss=loss.item())


    epoch_loss = running_loss / total_samples if total_samples > 0 else float('inf')
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    epoch_duration = time.time() - epoch_start_time
    return epoch_loss, epoch_acc, epoch_duration


# --- Main Script ---
if __name__ == '__main__':
    experiment_start_time = time.time()
    if not os.path.exists(os.path.dirname(STUDENT_MODEL_SAVE_PATH)) and os.path.dirname(STUDENT_MODEL_SAVE_PATH) != '':
        os.makedirs(os.path.dirname(STUDENT_MODEL_SAVE_PATH))
    if not os.path.exists(os.path.dirname(RESULTS_JSON_PATH)) and os.path.dirname(RESULTS_JSON_PATH) != '':
        os.makedirs(os.path.dirname(RESULTS_JSON_PATH))


    experiment_results = {
        "experiment_type": "knowledge_distillation",
        "config": {},
        "teacher_model_details": {},
        "student_distillation_results": {
            "student_architecture": STUDENT_ARCHITECTURE,
            "training_history": {
                'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_times': []
            },
            "best_student_model_path": STUDENT_MODEL_SAVE_PATH,
            "final_evaluation_metrics": None,
            "distillation_config_snapshot": {}
        },
        "completed_at": None,
        "total_experiment_time_seconds": None
    }

    experiment_results["config"] = {
        "DATA_DIR": DATA_DIR,
        "TEACHER_MODEL_PATH": TEACHER_MODEL_PATH,
        "STUDENT_MODEL_SAVE_PATH": STUDENT_MODEL_SAVE_PATH,
        "RESULTS_JSON_PATH": RESULTS_JSON_PATH,
        "STUDENT_ARCHITECTURE": STUDENT_ARCHITECTURE,
        "NUM_CLASSES_initial": NUM_CLASSES,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "INITIAL_STUDENT_LR": INITIAL_STUDENT_LR,
        "TEMPERATURE": TEMPERATURE,
        "ALPHA": ALPHA,
        "DEVICE": str(DEVICE),
        "NUM_WORKERS": NUM_WORKERS,
        "PIN_MEMORY": PIN_MEMORY,
        "NUM_INF_BATCHES_SPEED_TEST": NUM_INF_BATCHES_SPEED_TEST
    }
    experiment_results["student_distillation_results"]["distillation_config_snapshot"] = {
        "epochs_target": EPOCHS,
        "initial_learning_rate": INITIAL_STUDENT_LR,
        "temperature": TEMPERATURE,
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE # Training batch size
    }


    use_num_classes_from_data = (NUM_CLASSES == -1)
    train_loader, val_loader, test_loader_inf, actual_num_classes_from_data = get_data_loaders(
        DATA_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, num_classes_from_data=use_num_classes_from_data
    )

    if NUM_CLASSES == -1 and actual_num_classes_from_data > 0:
        NUM_CLASSES = actual_num_classes_from_data
    elif NUM_CLASSES == -1 and actual_num_classes_from_data == 0:
        print("ERROR: NUM_CLASSES not set and could not be determined. Exiting.")
        exit()
    elif NUM_CLASSES != actual_num_classes_from_data and actual_num_classes_from_data > 0 :
         print(f"WARNING: Manually set NUM_CLASSES ({NUM_CLASSES}) doesn't match data ({actual_num_classes_from_data}). Using manual.")

    if not train_loader or not val_loader or not test_loader_inf:
        print("Failed to load data. Exiting.")
        exit()

    experiment_results["config"]["NUM_CLASSES_final"] = NUM_CLASSES
    print(f"Final NUM_CLASSES used for models: {NUM_CLASSES}")

    # --- Teacher Model Evaluation ---
    print("\nLoading and Evaluating Teacher Model (ResNet50)...")
    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f"ERROR: Teacher model path not found: {TEACHER_MODEL_PATH}")
        exit()
    teacher_model = get_model('resnet50', NUM_CLASSES, pretrained=False)
    try:
        state_dict = torch.load(TEACHER_MODEL_PATH, map_location=DEVICE)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        teacher_model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading teacher model state_dict: {e}")
        exit()
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_val_loss, teacher_val_acc, teacher_inf_metrics = evaluate_model_from_pruning_script(
        teacher_model, criterion_hard, val_loader, DEVICE,
        measure_speed=True, test_loader_inf=test_loader_inf, num_inf_batches=NUM_INF_BATCHES_SPEED_TEST,
        eval_desc="Evaluating Teacher"
    )
    teacher_total_params = get_total_model_parameters(teacher_model)
    teacher_nonzero_params = count_effective_non_zero_parameters(teacher_model)
    teacher_parameter_counts = {"total_params": teacher_total_params, "non_zero_params": teacher_nonzero_params}
    teacher_size_mb = get_model_size_mb(teacher_model) # Using the pruning script's version

    experiment_results["teacher_model_details"] = {
        "model_path": TEACHER_MODEL_PATH,
        "architecture": "resnet50",
        "val_accuracy": teacher_val_acc,
        "val_loss": teacher_val_loss,
        "model_size_mb": teacher_size_mb,
        "parameter_counts": teacher_parameter_counts,
        "inference_metrics": teacher_inf_metrics
    }
    print(f"Teacher - Val Acc: {teacher_val_acc:.4f}, Val Loss: {teacher_val_loss:.4f}, Size: {teacher_size_mb:.2f}MB")
    if teacher_inf_metrics:
        print(f"  Teacher Inference: {teacher_inf_metrics.get('images_per_second', 0):.2f} IPS")


    # --- Student Model ---
    print(f"\nInitializing Student Model ({STUDENT_ARCHITECTURE}) - Pretrained: {STUDENT_PRETRAINED}...")
    student_model = get_model(STUDENT_ARCHITECTURE, NUM_CLASSES, pretrained=STUDENT_PRETRAINED)


    optimizer = optim.Adam(student_model.parameters(), lr=INITIAL_STUDENT_LR)

    # --- Learning Rate Scheduler ---
    print(f"Using StepLR scheduler: step_size={SCHEDULER_STEP_SIZE}, gamma={SCHEDULER_GAMMA}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # --- Training Loop for Distillation ---
    best_val_accuracy = 0.0
    best_student_model_wts = None
    history = experiment_results["student_distillation_results"]["training_history"]
    distillation_training_start_time = time.time()

    print(f"\nStarting Knowledge Distillation for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_duration = train_epoch_kd(
            student_model, teacher_model, train_loader, optimizer, TEMPERATURE, ALPHA, epoch, EPOCHS
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc, val_duration = validate_epoch_kd(
            student_model, criterion_hard, val_loader, DEVICE, epoch, EPOCHS
        )
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(train_duration + val_duration)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | Time: {train_duration+val_duration:.2f}s | LR: {optimizer.param_groups[0]['lr']:.1e}") # Added LR print

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_student_model_wts = copy.deepcopy(student_model.state_dict())
            torch.save(best_student_model_wts, STUDENT_MODEL_SAVE_PATH)
            print(f"  New best validation accuracy: {best_val_accuracy:.4f}. Model saved.")

        # Step the scheduler
        scheduler.step() # For StepLR, MultiStepLR, CosineAnnealingLR

    distillation_training_total_time = time.time() - distillation_training_start_time
    experiment_results["student_distillation_results"]["distillation_config_snapshot"]["total_training_time_seconds"] = distillation_training_total_time
    experiment_results["student_distillation_results"]["distillation_config_snapshot"]["epochs_completed"] = EPOCHS
    print(f"\n--- Distillation Training Complete ({distillation_training_total_time:.2f} seconds) ---")


    # --- Final Evaluation of Best Student Model ---
    print("\nEvaluating Best Student Model...")
    if best_student_model_wts is None and os.path.exists(STUDENT_MODEL_SAVE_PATH):
        print(f"Loading best student model from {STUDENT_MODEL_SAVE_PATH} as no new best model was found or training was short.")
        best_student_model_wts = torch.load(STUDENT_MODEL_SAVE_PATH, map_location=DEVICE)

    if best_student_model_wts is not None:
        final_student_model = get_model(STUDENT_ARCHITECTURE, NUM_CLASSES, pretrained=False)
        final_student_model.load_state_dict(best_student_model_wts)
        final_student_model.to(DEVICE)

        student_final_val_loss, student_final_val_acc, student_inf_metrics = evaluate_model_from_pruning_script(
            final_student_model, criterion_hard, val_loader, DEVICE,
            measure_speed=True, test_loader_inf=test_loader_inf, num_inf_batches=NUM_INF_BATCHES_SPEED_TEST,
            eval_desc="Evaluating Final Student"
        )
        student_total_params = get_total_model_parameters(final_student_model)
        student_nonzero_params = count_effective_non_zero_parameters(final_student_model)
        student_parameter_counts = {"total_params": student_total_params, "non_zero_params": student_nonzero_params}
        student_size_mb = get_model_size_mb(final_student_model)

        experiment_results["student_distillation_results"]["final_evaluation_metrics"] = {
            "val_accuracy": student_final_val_acc,
            "val_loss": student_final_val_loss,
            "model_size_mb": student_size_mb,
            "parameter_counts": student_parameter_counts,
            "inference_metrics": student_inf_metrics,
            "achieved_best_val_accuracy_during_training": best_val_accuracy
        }
        print(f"Best Student ({STUDENT_ARCHITECTURE}) - Val Acc: {student_final_val_acc:.4f}, Loss: {student_final_val_loss:.4f}, Size: {student_size_mb:.2f}MB")
        if student_inf_metrics:
            print(f"  Best Student Inference: {student_inf_metrics.get('images_per_second', 0):.2f} IPS")
    else:
        print("No best student model weights found to evaluate.")
        experiment_results["student_distillation_results"]["final_evaluation_metrics"] = "No model saved or training performed sufficiently."

    # --- Finalize and Save Results ---
    experiment_results["completed_at"] = datetime.datetime.now().isoformat()
    experiment_results["total_experiment_time_seconds"] = time.time() - experiment_start_time

    try:
        with open(RESULTS_JSON_PATH, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        print(f"\nExperiment results saved to {RESULTS_JSON_PATH}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")