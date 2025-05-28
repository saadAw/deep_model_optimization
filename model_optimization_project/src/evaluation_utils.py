import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
import io
from typing import Dict, Tuple, Any, Optional # Added Optional for logger

# Assuming TrainingLogger might be used for logging within these utils,
# though not strictly required by the current function signatures.
# from .logger_utils import TrainingLogger # Optional: if logging is needed directly here

def validate_model(
    model: nn.Module, 
    data_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    logger: Optional[Any] = None # Optional logger for progress
) -> Tuple[float, float]:
    """
    Validate the model on a data loader.

    Args:
        model (nn.Module): The PyTorch model to validate.
        data_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to perform validation on.
        logger (Optional[Any]): An optional logger object with an `info` method.

    Returns:
        Tuple[float, float]: Average validation loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)  # Accumulate loss scaled by batch size
            _, predicted = outputs.max(1)  # Get the index of the max log-probability
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if logger and (batch_idx + 1) % 100 == 0: # Log every 100 batches if logger provided
                 logger.info(f"Validation Batch {batch_idx+1}/{len(data_loader)}: Current Batch Loss: {loss.item():.4f}")


    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    if logger:
        logger.info(f"Validation Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} ({correct}/{total})")

    return avg_loss, accuracy


def measure_inference_metrics(
    model: nn.Module, 
    val_loader: DataLoader,  # Changed from data_loader to val_loader to match usage
    device: torch.device, 
    num_warmup_batches: int = 10, # Increased default warmup
    num_timing_batches: int = 100, # Increased default timing batches
    logger: Optional[Any] = None # Optional logger
) -> Dict[str, float]:
    """
    Measure model inference speed and latency on the specified device.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        val_loader (DataLoader): DataLoader for the validation/test set.
        device (torch.device): The device for inference.
        num_warmup_batches (int): Number of initial batches for GPU warmup.
        num_timing_batches (int): Number of batches to run for actual timing.
        logger (Optional[Any]): An optional logger object with an `info` and `warning` method.

    Returns:
        Dict[str, float]: A dictionary containing:
            - 'images_per_second': Throughput.
            - 'latency_ms_per_image': Average latency per image.
            - 'total_images_measured': Total number of images processed during timing.
            - 'total_time_seconds': Total time taken for processing timed batches.
    """
    if logger:
        logger.info(f"Measuring inference metrics with {num_warmup_batches} warmup batches and {num_timing_batches} timing batches.")
    
    model.eval() # Set model to evaluation mode

    # Create a new DataLoader for timing to ensure consistent batch_size and no shuffling.
    # Using num_workers=0 can sometimes give more stable results for timing by avoiding IPC overhead.
    # However, if the original val_loader uses num_workers > 0, this might not reflect real-world throughput.
    # For this generic utility, we'll use a dedicated timing_loader.
    try:
        timing_loader = DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size, # Use same batch size as val_loader
            shuffle=False, # Crucial for reproducibility and stable measurement
            num_workers=0, # For potentially more stable timing
            pin_memory=val_loader.pin_memory # Match pin_memory setting
        )
    except Exception as e:
        if logger:
            logger.warning(f"Failed to create dedicated timing DataLoader: {e}. Using provided val_loader directly.")
        timing_loader = val_loader # Fallback to using the provided loader


    total_batches_available = len(timing_loader)
    if total_batches_available < num_warmup_batches + num_timing_batches:
        if logger:
            logger.warning(
                f"Not enough batches for desired warmup/timing ({total_batches_available} available). "
                f"Adjusting num_timing_batches."
            )
        # Prioritize warmup. Reduce timing batches if necessary.
        num_timing_batches = max(0, total_batches_available - num_warmup_batches)

    if num_timing_batches == 0:
        if logger:
            logger.warning("Cannot perform timing (0 timing batches). Returning zero metrics.")
        return {
            'images_per_second': 0.0, 'latency_ms_per_image': 0.0,
            'total_images_measured': 0, 'total_time_seconds': 0.0
        }

    if logger:
        logger.info(f"Actual timing batches: {num_timing_batches}, Warmup batches: {num_warmup_batches}")

    timing_iterator = iter(timing_loader)

    with torch.no_grad():
        # Warmup phase
        if logger and num_warmup_batches > 0:
            logger.info("Starting warmup...")
        for i in range(num_warmup_batches):
            try:
                inputs, _ = next(timing_iterator)
                inputs = inputs.to(device)
                _ = model(inputs)
            except StopIteration:
                if logger:
                    logger.warning(f"Ran out of batches during warmup at batch {i+1}. Proceeding with timing if possible.")
                break # Should not happen if total_batches_available was checked, but good practice

        # Synchronize if on CUDA device before starting timer
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Timing phase
        if logger:
            logger.info("Starting timing measurement...")
        total_images_processed = 0
        start_time = time.perf_counter() # Use perf_counter for more precise timing

        for i in range(num_timing_batches):
            try:
                inputs, _ = next(timing_iterator)
                inputs = inputs.to(device)
                _ = model(inputs)
                total_images_processed += inputs.size(0)
            except StopIteration:
                if logger:
                    logger.warning(f"Ran out of batches during timing at batch {i+1}. Results will be based on completed batches.")
                break # Should not happen if checks are correct

        # Synchronize if on CUDA device before stopping timer
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        total_inference_time_seconds = end_time - start_time

    if total_images_processed == 0 or total_inference_time_seconds <= 0:
        if logger:
            logger.warning("No images processed or zero time recorded during timing. Returning zero metrics.")
        images_per_sec = 0.0
        latency_ms_per_img = 0.0
    else:
        images_per_sec = total_images_processed / total_inference_time_seconds
        latency_ms_per_img = (total_inference_time_seconds / total_images_processed) * 1000.0
        if logger:
            logger.info(f"Timing complete: Processed {total_images_processed} images in {total_inference_time_seconds:.3f}s.")

    return {
        'images_per_second': float(images_per_sec),
        'latency_ms_per_image': float(latency_ms_per_img),
        'total_images_measured': total_images_processed,
        'total_time_seconds': float(total_inference_time_seconds)
    }


def calculate_model_size_mb_eval(model: nn.Module) -> float:
    """
    Calculates the size of a model's state_dict in megabytes (MB).
    This version takes the model object directly.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        float: The size of the model's state_dict in MB.
    """
    # Ensure model is on CPU for consistent size calculation and to avoid GPU memory usage
    # This is a good practice, though state_dict itself doesn't store device info in a way that affects size.
    # original_device = next(model.parameters()).device # Get current device
    # model.to('cpu')

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer) # Save state_dict to the buffer
    model_size_bytes = buffer.getbuffer().nbytes
    model_size_mb = model_size_bytes / (1024 ** 2) # Convert bytes to MB
    buffer.close()

    # model.to(original_device) # Restore model to its original device
    return model_size_mb


def save_final_results(results: Dict[str, Any], save_dir: Path, filename: str = "final_metrics.json"):
    """
    Saves a dictionary of results to a JSON file.

    Args:
        results (Dict[str, Any]): Dictionary containing the results to save.
        save_dir (Path): The directory where the results file will be saved.
        filename (str): The name of the JSON file.
    """
    save_dir.mkdir(parents=True, exist_ok=True) # Ensure the save directory exists
    results_path = save_dir / filename

    try:
        with open(results_path, 'w') as f:
            # Ensure Path objects in results are converted to strings for JSON serialization
            def convert_path_to_str(obj):
                if isinstance(obj, Path):
                    return str(obj)
                # Handle nested dictionaries
                if isinstance(obj, dict):
                    return {k: convert_path_to_str(v) for k, v in obj.items()}
                # Handle lists of items (e.g., list of Paths or dicts)
                if isinstance(obj, list):
                    return [convert_path_to_str(item) for item in obj]
                return obj

            results_serializable = convert_path_to_str(results)
            json.dump(results_serializable, f, indent=2)
        print(f"Final results saved to: {results_path}") # Simple print for user feedback
    except IOError as e:
        print(f"ERROR: Failed to save final results to {results_path}: {e}")
    except TypeError as e:
        print(f"ERROR: Failed to serialize results to JSON. Check for non-serializable types: {e}")


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing evaluation_utils.py ---")

    # Dummy components for testing
    from torchvision import models
    from .logger_utils import TrainingLogger # For testing logger integration
    
    test_save_dir = Path("./temp_eval_utils_test_run")
    test_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup a dummy logger
    test_logger = TrainingLogger(save_dir=test_save_dir, log_file_name="eval_utils_test.log")
    test_logger.info("Starting evaluation_utils tests.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_logger.info(f"Using device: {device}")

    # Dummy Model, Criterion, DataLoader
    model = models.resnet18(weights=None).to(device) # Using weights=None for modern torchvision
    criterion = nn.CrossEntropyLoss()
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, num_classes=10, img_size=(3, 224, 224)):
            self.num_samples = num_samples
            self.data = torch.randn(num_samples, *img_size)
            self.targets = torch.randint(0, num_classes, (num_samples,))
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    val_dataset = DummyDataset(num_samples=500) # Larger dataset for more stable timing
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0) # num_workers=0 for timing
    test_logger.info(f"Dummy DataLoader: {len(val_loader)} batches, {len(val_dataset)} samples.")


    # 1. Test validate_model
    test_logger.info("\n--- Testing validate_model ---")
    try:
        val_loss, val_acc = validate_model(model, val_loader, criterion, device, logger=test_logger)
        test_logger.info(f"validate_model - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        assert isinstance(val_loss, float) and val_loss >= 0
        assert isinstance(val_acc, float) and 0 <= val_acc <= 1
    except Exception as e:
        test_logger.error(f"Error during validate_model test: {e}", exc_info=True)

    # 2. Test measure_inference_metrics
    test_logger.info("\n--- Testing measure_inference_metrics ---")
    try:
        # Reduce batches for quicker test, but keep them reasonable
        metrics = measure_inference_metrics(model, val_loader, device, 
                                            num_warmup_batches=2, num_timing_batches=5, logger=test_logger)
        test_logger.info(f"measure_inference_metrics - Metrics: {json.dumps(metrics, indent=2)}")
        assert 'images_per_second' in metrics
        assert 'latency_ms_per_image' in metrics
        assert metrics['total_images_measured'] > 0 or (metrics['total_images_measured'] == 0 and metrics['total_time_seconds'] == 0)

        # Test with very few batches to check warning paths
        short_val_dataset = DummyDataset(num_samples=val_loader.batch_size * 2) # e.g., 2 batches
        short_val_loader = DataLoader(short_val_dataset, batch_size=val_loader.batch_size)
        test_logger.info("Testing measure_inference_metrics with insufficient batches...")
        metrics_short = measure_inference_metrics(model, short_val_loader, device, 
                                                  num_warmup_batches=5, num_timing_batches=10, logger=test_logger)
        test_logger.info(f"measure_inference_metrics (short) - Metrics: {json.dumps(metrics_short, indent=2)}")
        assert metrics_short['total_images_measured'] == 0 # Should result in 0 timing batches

    except Exception as e:
        test_logger.error(f"Error during measure_inference_metrics test: {e}", exc_info=True)

    # 3. Test calculate_model_size_mb_eval
    test_logger.info("\n--- Testing calculate_model_size_mb_eval ---")
    try:
        model_size = calculate_model_size_mb_eval(model)
        test_logger.info(f"calculate_model_size_mb_eval - Model Size: {model_size:.2f} MB")
        assert isinstance(model_size, float) and model_size > 0
    except Exception as e:
        test_logger.error(f"Error during calculate_model_size_mb_eval test: {e}", exc_info=True)

    # 4. Test save_final_results
    test_logger.info("\n--- Testing save_final_results ---")
    try:
        dummy_results = {
            'model_name': 'resnet18_test',
            'accuracy': val_acc if 'val_acc' in locals() else 0.0,
            'loss': val_loss if 'val_loss' in locals() else 0.0,
            'size_mb': model_size if 'model_size' in locals() else 0.0,
            'inference_speed_ips': metrics.get('images_per_second', 0.0) if 'metrics' in locals() else 0.0,
            'config_params': {'lr': 0.001, 'batch_size': 32, 'data_path': Path('/tmp/data')}
        }
        save_final_results(dummy_results, test_save_dir, filename="test_results.json")
        
        # Verify file was created
        assert (test_save_dir / "test_results.json").exists()
        test_logger.info(f"save_final_results - Test results saved to {test_save_dir / 'test_results.json'}")
        
        # Verify Path object serialization
        with open(test_save_dir / "test_results.json", 'r') as f:
            loaded_results = json.load(f)
        assert loaded_results['config_params']['data_path'] == '/tmp/data' # Should be string

    except Exception as e:
        test_logger.error(f"Error during save_final_results test: {e}", exc_info=True)

    test_logger.info("\nAll evaluation_utils tests finished.")
    print(f"\nExample finished. Check logs and results in {test_save_dir.resolve()}")
    
    # Note: To fully clean up, the temp_eval_utils_test_run directory can be removed.
    # import shutil
    # shutil.rmtree(test_save_dir)
    # print(f"Cleaned up directory: {test_save_dir.resolve()}")
