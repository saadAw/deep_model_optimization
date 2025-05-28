import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, Any, List, Tuple, Type # Added List, Tuple, Type

# Assuming model_utils is in the same src directory
from .model_utils import count_model_parameters


def apply_global_unstructured_pruning(
    model: nn.Module, 
    sparsity_rate: float, 
    pruning_method: Type[prune.BasePruningMethod] = prune.L1Unstructured, # Use Type for class
    prunable_types: List[Type[nn.Module]] = [nn.Conv2d, nn.Linear], # Default prunable layer types
    logger: Optional[Any] = None
):
    """
    Applies global unstructured pruning to specified layer types in the model.

    Args:
        model (nn.Module): The PyTorch model to prune.
        sparsity_rate (float): The target sparsity rate (0.0 to 1.0).
        pruning_method (Type[prune.BasePruningMethod]): The pruning method to use 
            (e.g., prune.L1Unstructured, prune.RandomUnstructured).
        prunable_types (List[Type[nn.Module]]): A list of nn.Module types to consider for pruning.
        logger (Optional[Any]): An optional logger object with an `info` and `warning` method.
    """
    if not (0.0 <= sparsity_rate <= 1.0):
        raise ValueError(f"Sparsity rate must be between 0.0 and 1.0, got {sparsity_rate}")

    if logger:
        logger.info(
            f"Applying global unstructured pruning with method {pruning_method.__name__} "
            f"to {sparsity_rate*100:.2f}% sparsity for layer types: {[t.__name__ for t in prunable_types]}."
        )

    parameters_to_prune: List[Tuple[nn.Module, str]] = []
    for module in model.modules(): # Iterate through all modules recursively
        # Check if the module type is in the list of prunable_types
        if any(isinstance(module, t) for t in prunable_types):
            # Check for 'weight' parameter
            if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                parameters_to_prune.append((module, 'weight'))
            # Optionally, check for 'bias' parameter if you want to prune biases too
            # if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
            #     parameters_to_prune.append((module, 'bias'))

    if not parameters_to_prune:
        if logger:
            logger.warning("No prunable parameters found (Conv2d or Linear layers with 'weight'). Pruning not applied.")
        return

    # Ensure sparsity_rate is not exactly 1.0 to avoid potential issues with some pruning methods
    # or completely zeroing out all weights if that's not intended.
    # PyTorch's global_unstructured handles amount=1.0 by zeroing all weights.
    amount_to_prune = min(sparsity_rate, 1.0 - 1e-7) if sparsity_rate == 1.0 else sparsity_rate
    
    if amount_to_prune > 0: # Only apply if there's a non-zero amount to prune
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=amount_to_prune,
        )
        if logger:
            num_pruned_tensors = len(parameters_to_prune)
            logger.info(f"Global unstructured pruning applied to {num_pruned_tensors} parameter tensor(s).")
    else:
        if logger:
            logger.info("Sparsity rate is 0.0. No pruning applied.")


def remove_pruning_reparametrization(model: nn.Module, logger: Optional[Any] = None):
    """
    Removes pruning reparameterization (masks) from the model, making pruning permanent.
    This iterates through all modules and removes the 'weight' and 'bias' pruning hooks if present.

    Args:
        model (nn.Module): The PyTorch model.
        logger (Optional[Any]): An optional logger object.
    """
    if logger:
        logger.info("Attempting to remove pruning reparameterization (masks)...")
    
    removed_hooks_count = 0
    for module in model.modules():
        # Check for 'weight' pruning
        if prune.is_pruned(module) and hasattr(module, 'weight_mask'): # More specific check
            try:
                prune.remove(module, 'weight')
                removed_hooks_count += 1
                if logger:
                    logger.debug(f"Removed 'weight' pruning mask from module: {type(module).__name__}")
            except Exception as e: # Catching generic exception is broad, but prune.remove can raise various things
                if logger:
                    logger.warning(f"Could not remove 'weight' mask from {type(module).__name__}: {e}")
        
        # Check for 'bias' pruning (if you ever prune biases)
        if hasattr(module, 'bias') and module.bias is not None and \
           prune.is_pruned(module) and hasattr(module, 'bias_mask'):
            try:
                prune.remove(module, 'bias')
                removed_hooks_count += 1
                if logger:
                    logger.debug(f"Removed 'bias' pruning mask from module: {type(module).__name__}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not remove 'bias' mask from {type(module).__name__}: {e}")
    
    if logger:
        if removed_hooks_count > 0:
            logger.info(f"Successfully removed pruning reparameterization from {removed_hooks_count} parameter(s) across modules.")
        else:
            logger.info("No pruning reparameterization found or removed from the model.")


def calculate_sparsity(model: nn.Module, include_bias: bool = False) -> float:
    """
    Calculates the actual sparsity of a model.
    Sparsity = (Total Zero Parameters) / (Total Parameters)

    Args:
        model (nn.Module): The PyTorch model.
        include_bias (bool): Whether to include bias parameters in the calculation. Default False.

    Returns:
        float: The actual sparsity of the model (0.0 to 1.0).
    """
    param_counts = count_model_parameters(model=model, count_bias=include_bias) # Pass include_bias
    total_params = param_counts['total_params_wb' if include_bias else 'total_params_wo_bias']
    non_zero_params = param_counts['non_zero_params_wb' if include_bias else 'non_zero_params_wo_bias']
    
    if total_params == 0: # Avoid division by zero if model has no parameters or specified types
        return 0.0
        
    sparsity = (total_params - non_zero_params) / total_params
    return sparsity


def convert_to_sparse_tensor_model(
    model: nn.Module, 
    prunable_types: List[Type[nn.Module]] = [nn.Conv2d, nn.Linear],
    logger: Optional[Any] = None
):
    """
    Converts weights of specified layer types (e.g., Conv2d, Linear) in a model 
    to PyTorch's sparse COO tensor format if they are significantly sparse.
    This is typically done after pruning and removing reparameterization.

    Note: This operation is IN-PLACE.

    Args:
        model (nn.Module): The PyTorch model (ideally pruned and masks removed).
        prunable_types (List[Type[nn.Module]]): List of layer types whose weights to convert.
        logger (Optional[Any]): An optional logger.
    """
    if logger:
        logger.info(f"Attempting to convert weights of types {[t.__name__ for t in prunable_types]} to sparse COO format.")

    converted_count = 0
    skipped_count = 0
    for module in model.modules():
        if any(isinstance(module, t) for t in prunable_types):
            if hasattr(module, 'weight') and module.weight is not None:
                weight_tensor = module.weight.data
                if not weight_tensor.is_sparse: # Only convert if not already sparse
                    # Check sparsity of the weight tensor itself
                    num_zeros = torch.sum(weight_tensor == 0).item()
                    total_elements = weight_tensor.numel()
                    tensor_sparsity = num_zeros / total_elements if total_elements > 0 else 0

                    # Heuristic: Only convert if tensor is reasonably sparse (e.g., >50%)
                    # and not too small, as sparse tensors have overhead.
                    # This threshold is arbitrary and might need tuning.
                    if tensor_sparsity > 0.5 and total_elements > 100: 
                        try:
                            module.weight.data = weight_tensor.to_sparse_coo()
                            converted_count += 1
                            if logger:
                                logger.debug(f"Converted weight of {type(module).__name__} to sparse COO (sparsity: {tensor_sparsity*100:.2f}%).")
                        except RuntimeError as e: # PyTorch might raise RuntimeError for non-convertible layouts etc.
                             if logger:
                                logger.warning(f"Could not convert weight of {type(module).__name__} to sparse: {e}")
                             skipped_count +=1
                    else:
                        if logger:
                            logger.debug(
                                f"Skipped converting weight of {type(module).__name__} due to low sparsity ({tensor_sparsity*100:.2f}%) "
                                f"or small size ({total_elements} elements)."
                            )
                        skipped_count += 1
                else:
                    if logger:
                        logger.debug(f"Weight of {type(module).__name__} is already sparse. Skipping.")
                    skipped_count += 1
    
    if logger:
        logger.info(f"Sparse tensor conversion: {converted_count} weights converted, {skipped_count} weights skipped/already_sparse.")


# --- Example Usage ---
if __name__ == '__main__':
    from torchvision import models
    from .logger_utils import TrainingLogger # For testing logger integration
    
    test_save_dir = Path("./temp_pruning_utils_test_run")
    test_save_dir.mkdir(parents=True, exist_ok=True)
    
    test_logger = TrainingLogger(save_dir=test_save_dir, log_file_name="pruning_utils_test.log")
    test_logger.info("--- Starting pruning_utils.py tests ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_logger.info(f"Using device: {device}")

    # --- Test apply_global_unstructured_pruning & calculate_sparsity ---
    test_logger.info("\n--- Testing apply_global_unstructured_pruning & calculate_sparsity ---")
    model_to_prune = models.resnet18(weights=None).to(device)
    initial_sparsity = calculate_sparsity(model_to_prune)
    test_logger.info(f"Initial model sparsity: {initial_sparsity*100:.2f}%")
    assert initial_sparsity == 0.0, "Fresh model should have 0.0 sparsity."
    
    # Count params before
    params_before = count_model_parameters(model_to_prune)
    test_logger.info(f"Params before pruning: Total (w/o bias)={params_before['total_params_wo_bias']:,}, Non-Zero (w/o bias)={params_before['non_zero_params_wo_bias']:,}")


    target_sparsity_rate = 0.5
    apply_global_unstructured_pruning(model_to_prune, target_sparsity_rate, logger=test_logger)
    
    # Sparsity calculation after pruning (masks are applied, zeros are "logical")
    # Note: count_model_parameters for a masked model counts the underlying non-zero values.
    # The `prune` module effectively makes some weights zero.
    sparsity_after_masking = calculate_sparsity(model_to_prune)
    test_logger.info(f"Sparsity after masking (target {target_sparsity_rate*100:.1f}%): {sparsity_after_masking*100:.2f}%")
    
    params_masked = count_model_parameters(model_to_prune)
    test_logger.info(f"Params after masking: Total (w/o bias)={params_masked['total_params_wo_bias']:,}, Non-Zero (w/o bias)={params_masked['non_zero_params_wo_bias']:,}")
    
    # Check if achieved sparsity is close to target (can be slightly off due to discrete nature of weights)
    # This assertion is tricky because global pruning might not hit the exact rate perfectly.
    # For L1Unstructured, it should be quite close for large models.
    assert abs(sparsity_after_masking - target_sparsity_rate) < 0.05, \
        f"Achieved sparsity {sparsity_after_masking:.3f} is not close to target {target_sparsity_rate:.3f}"


    # --- Test remove_pruning_reparametrization ---
    test_logger.info("\n--- Testing remove_pruning_reparametrization ---")
    remove_pruning_reparametrization(model_to_prune, logger=test_logger)
    
    # Verify masks are removed (prune.is_pruned should be False for pruned modules)
    is_still_pruned_somewhere = False
    for module in model_to_prune.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                is_still_pruned_somewhere = True
                test_logger.error(f"Module {type(module).__name__} is still pruned after removal call.")
                break
    assert not is_still_pruned_somewhere, "Pruning masks were not removed properly."
    test_logger.info("Successfully verified masks are removed.")

    sparsity_after_removal = calculate_sparsity(model_to_prune)
    test_logger.info(f"Sparsity after mask removal: {sparsity_after_removal*100:.2f}%")
    # Sparsity should be the same as with masks, as `remove` makes the zeros permanent.
    assert abs(sparsity_after_removal - target_sparsity_rate) < 0.05, \
         f"Sparsity after removal {sparsity_after_removal:.3f} differs too much from target {target_sparsity_rate:.3f}"

    # --- Test convert_to_sparse_tensor_model ---
    test_logger.info("\n--- Testing convert_to_sparse_tensor_model ---")
    # Create a new, heavily pruned model for this test
    sparse_test_model = models.resnet18(weights=None).to(device)
    high_sparsity_rate = 0.90 # High sparsity to ensure some layers are sparse enough
    apply_global_unstructured_pruning(sparse_test_model, high_sparsity_rate, logger=test_logger)
    remove_pruning_reparametrization(sparse_test_model, logger=test_logger) # Remove masks first
    
    convert_to_sparse_tensor_model(sparse_test_model, logger=test_logger)
    
    sparse_tensors_found = 0
    for name, param in sparse_test_model.named_parameters():
        if param.is_sparse:
            sparse_tensors_found +=1
            test_logger.info(f"Parameter '{name}' is sparse. Shape: {param.shape}, Density: {param.to_dense().count_nonzero()/param.numel():.3f}")
    
    # This assertion depends on the model structure and pruning.
    # For ResNet18 and 90% sparsity, at least some layers should become sparse.
    assert sparse_tensors_found > 0, "Expected some weights to be converted to sparse format, but none found."
    test_logger.info(f"Found {sparse_tensors_found} sparse parameter tensors in the model.")


    test_logger.info("\nAll pruning_utils tests finished.")
    print(f"\nExample finished. Check logs in {test_save_dir.resolve()}")
    
    # Cleanup (optional)
    # import shutil
    # shutil.rmtree(test_save_dir)
    # print(f"Cleaned up directory: {test_save_dir.resolve()}")
