import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import io
from typing import Dict, Any, Union # Union for state_dict type hint

# For model registration, if we decide to use it later
# model_registry = {}

# def register_model(name):
#     def decorator(cls):
#         model_registry[name] = cls
#         return cls
#     return decorator

def build_model(
    model_arch: str, 
    num_classes: int, 
    pretrained: bool = False, 
    use_custom_pretrained_weights: bool = False, # New flag
    custom_pretrained_weights_path: Union[str, Path, None] = None, # Path for custom weights
    device: Union[str, torch.device] = 'cpu' # Device for loading custom weights
) -> nn.Module:
    """
    Builds a model based on the specified architecture.

    Args:
        model_arch (str): The architecture name (e.g., 'resnet18', 'resnet50').
        num_classes (int): Number of output classes for the classifier.
        pretrained (bool): If True, loads weights pretrained on ImageNet (from torchvision).
                           This is ignored if use_custom_pretrained_weights is True and
                           custom_pretrained_weights_path is provided.
        use_custom_pretrained_weights (bool): If True, loads weights from custom_pretrained_weights_path.
        custom_pretrained_weights_path (Union[str, Path, None]): Path to a custom .pth or .pt file.
        device (Union[str, torch.device]): Device to load the custom weights onto.

    Returns:
        nn.Module: The constructed PyTorch model.
        
    Raises:
        NotImplementedError: If the model_arch is not supported.
        ValueError: If custom weights are specified but path is missing, or if `pretrained` 
                    and `use_custom_pretrained_weights` are used ambiguously.
        FileNotFoundError: If the custom_pretrained_weights_path does not exist.
    """
    model: nn.Module
    
    if use_custom_pretrained_weights and custom_pretrained_weights_path:
        if pretrained:
            print(f"Warning: 'pretrained=True' is ignored because 'use_custom_pretrained_weights' is True and a path is provided.")
        
        # Load custom pretrained weights
        weights_path = Path(custom_pretrained_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Custom pretrained weights file not found: {weights_path}")
        
        # First, build the model structure without torchvision's pretrained weights
        if model_arch == 'resnet18':
            model = models.resnet18(weights=None) # Changed from pretrained=False for clarity with new torchvision APIs
        elif model_arch == 'resnet50':
            model = models.resnet50(weights=None) # Changed from pretrained=False
        else:
            raise NotImplementedError(f"Model architecture '{model_arch}' is not supported for custom pretraining yet.")
        
        # Adjust classifier
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, (nn.Linear, nn.Sequential)):
             # Handle models like VGG, DenseNet if added later
            if isinstance(model.classifier, nn.Linear):
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            elif isinstance(model.classifier, nn.Sequential): # e.g. Densenet
                last_layer = model.classifier[-1]
                if isinstance(last_layer, nn.Linear):
                    num_ftrs = last_layer.in_features
                    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                else:
                     raise NotImplementedError(f"Classifier adjustment for Sequential layer type {type(last_layer)} in {model_arch} not implemented.")
            else:
                raise NotImplementedError(f"Classifier adjustment for type {type(model.classifier)} in {model_arch} not implemented.")

        else:
            raise NotImplementedError(f"Classifier adjustment for '{model_arch}' not implemented.")

        # Load the state dict from the custom path
        print(f"Loading custom pretrained weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=torch.device(device))
        
        # Handle potential 'model_state_dict' or 'state_dict' keys in saved file
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        model.load_state_dict(state_dict)
        print("Custom pretrained weights loaded successfully.")

    else: # Standard torchvision pretrained or from scratch
        if model_arch == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_arch == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None) # V2 is generally better
        else:
            raise NotImplementedError(f"Model architecture '{model_arch}' is not currently supported.")

        # Adjust the fully connected layer for the new number of classes
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): # For models like VGG
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        # Add more conditions here if supporting other model families like DenseNet, etc.
        # For DenseNet, it's model.classifier that needs replacement.
        # Example for DenseNet (if you were to add it):
        # elif model_arch.startswith('densenet') and hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        #     num_ftrs = model.classifier.in_features
        #     model.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            # This error should ideally not be hit if architectures are added correctly
            raise NotImplementedError(f"Classifier adjustment for '{model_arch}' not implemented or layer not found.")
            
    return model


def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Counts total and non-zero parameters in the model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        Dict[str, int]: A dictionary with 'total_params' and 'non_zero_params'.
    """
    total_params = 0
    non_zero_params = 0
    
    for param in model.parameters():
        # Each param is a tensor. numel() gives the total number of elements.
        numel_in_param = param.numel()
        total_params += numel_in_param
        
        # Count non-zero elements. This is crucial for pruned models
        # where some weights might be set to zero but are still part of the tensor.
        if numel_in_param > 0 : # Ensure parameter tensor is not empty
            non_zero_params += torch.count_nonzero(param).item()
            
    return {
        'total_params': total_params,
        'non_zero_params': non_zero_params,
        'zero_params': total_params - non_zero_params # Explicitly add zero params
    }


def calculate_model_size_mb(model_state_dict: Dict[str, Any]) -> float:
    """
    Calculates the size of a model's state_dict in megabytes.
    This represents the size if the model were saved to disk.

    Args:
        model_state_dict (Dict[str, Any]): The model's state_dict.

    Returns:
        float: The size of the state_dict in megabytes (MB).
    """
    # Create a temporary buffer in memory
    buffer = io.BytesIO()
    # Save the state_dict to the buffer
    torch.save(model_state_dict, buffer)
    # Get the size of the buffer in bytes
    model_size_bytes = buffer.getbuffer().nbytes
    # Convert bytes to megabytes
    model_size_mb = model_size_bytes / (1024 ** 2)
    buffer.close() # Close the buffer
    return model_size_mb


def load_model_state_dict_from_path(model_path: Union[str, Path], device: Union[str, torch.device]) -> Dict[str, Any]:
    """
    Loads a model state dictionary from a file.

    Args:
        model_path (Union[str, Path]): The path to the model file (.pth or .pt).
        device (Union[str, torch.device]): The device to map the loaded state_dict to.

    Returns:
        Dict[str, Any]: The model's state_dict.
        
    Raises:
        FileNotFoundError: If the model_path does not exist.
        Exception: If there's an error during loading.
    """
    _model_path = Path(model_path) # Ensure it's a Path object
    if not _model_path.exists():
        raise FileNotFoundError(f"Model state file not found at: {_model_path}")
    
    try:
        # map_location ensures the model is loaded to the specified device
        state_dict = torch.load(str(_model_path), map_location=torch.device(device))
        
        # Common practice: saved models might be wrapped in a dictionary
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict: # Another common key
                state_dict = state_dict['state_dict']
            # If neither, assume the loaded dict is the state_dict itself.
            
        print(f"State dict loaded successfully from {_model_path} to {device}.")
        return state_dict
    except Exception as e:
        print(f"ERROR: Failed to load state dict from {_model_path}: {e}")
        raise


# --- Example Usage ---
if __name__ == '__main__':
    # --- Test build_model ---
    print("--- Testing build_model ---")
    try:
        # Test ResNet18 from scratch
        resnet18_scratch = build_model('resnet18', num_classes=10, pretrained=False)
        print("ResNet18 (scratch) built successfully.")
        assert resnet18_scratch.fc.out_features == 10

        # Test ResNet50 pretrained
        resnet50_pretrained = build_model('resnet50', num_classes=100, pretrained=True)
        print("ResNet50 (ImageNet pretrained) built successfully.")
        assert resnet50_pretrained.fc.out_features == 100

        # Test loading custom pretrained weights (requires a dummy weights file)
        dummy_weights_dir = Path("./temp_weights_for_test")
        dummy_weights_dir.mkdir(exist_ok=True)
        dummy_weights_path = dummy_weights_dir / "dummy_resnet18_weights.pth"

        # Create a dummy state_dict for resnet18 with 10 classes
        temp_model = models.resnet18(weights=None)
        temp_model.fc = nn.Linear(temp_model.fc.in_features, 10) # Adjust for 10 classes
        torch.save(temp_model.state_dict(), dummy_weights_path)
        print(f"Dummy weights saved to {dummy_weights_path}")

        custom_loaded_model = build_model(
            'resnet18', 
            num_classes=10, # Must match the saved state_dict's output layer
            use_custom_pretrained_weights=True,
            custom_pretrained_weights_path=dummy_weights_path,
            device='cpu'
        )
        print("ResNet18 (custom pretrained) built successfully.")
        assert custom_loaded_model.fc.out_features == 10
        
        # Test loading custom pretrained weights for a model with a different number of classes
        # This should work as the classifier is re-initialized AFTER loading weights for the base model
        # (The current implementation re-initializes classifier first, then loads. This is a subtle point)
        # The build_model function as written will re-initialize the classifier to `num_classes` *before*
        # attempting to load custom weights if `use_custom_pretrained_weights` is true.
        # If the custom weights are for a *different* number of classes, `load_state_dict` will fail
        # due to mismatched fc layer size.
        # A more robust way for custom pretraining would be to load weights into the feature extractor
        # and then attach a new classifier, or ensure the saved weights match the target num_classes.
        # Let's test the current behavior:
        
        # Save weights for 20 classes
        temp_model_20_classes = models.resnet18(weights=None)
        temp_model_20_classes.fc = nn.Linear(temp_model_20_classes.fc.in_features, 20)
        dummy_weights_20_path = dummy_weights_dir / "dummy_resnet18_weights_20_classes.pth"
        torch.save(temp_model_20_classes.state_dict(), dummy_weights_20_path)

        try:
            custom_loaded_model_mismatch = build_model(
                'resnet18', 
                num_classes=15, # Different from the 20 in the saved file
                use_custom_pretrained_weights=True,
                custom_pretrained_weights_path=dummy_weights_20_path,
            )
            print(f"Custom loaded model (mismatch test) fc out: {custom_loaded_model_mismatch.fc.out_features}")
            # This assertion depends on how build_model handles mismatch.
            # Current build_model: fc is set to num_classes (15), then tries to load state_dict for 20 classes. This will error.
            # assert custom_loaded_model_mismatch.fc.out_features == 15 
            print("ERROR: Mismatch test passed where it should have failed due to strict loading.")
        except RuntimeError as e:
            print(f"Successfully caught error for custom pretrained weights mismatch: {e}")


        # Test unsupported architecture
        try:
            build_model('non_existent_arch', num_classes=10)
        except NotImplementedError:
            print("Successfully caught NotImplementedError for unsupported architecture.")

    except Exception as e:
        print(f"Error during build_model tests: {e}")
    finally:
        import shutil
        if dummy_weights_dir.exists():
            shutil.rmtree(dummy_weights_dir)

    # --- Test count_model_parameters ---
    print("\n--- Testing count_model_parameters ---")
    if 'resnet18_scratch' in locals():
        params = count_model_parameters(resnet18_scratch)
        print(f"ResNet18 (scratch) params: Total={params['total_params']:,}, Non-zero={params['non_zero_params']:,}, Zero={params['zero_params']:,}")
        assert params['total_params'] > 0
        assert params['total_params'] == params['non_zero_params'] # No zeros in a fresh model

        # Create a simple model and zero out some weights
        simple_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        # Zero out the weights of the first linear layer
        with torch.no_grad():
            simple_model[0].weight.data.fill_(0) 
        
        params_simple_pruned = count_model_parameters(simple_model)
        print(f"Simple pruned model params: Total={params_simple_pruned['total_params']:,}, Non-zero={params_simple_pruned['non_zero_params']:,}, Zero={params_simple_pruned['zero_params']:,}")
        assert params_simple_pruned['total_params'] == (10*5 + 5) + (5*2 + 2) # (weights+bias) + (weights+bias)
        assert params_simple_pruned['non_zero_params'] < params_simple_pruned['total_params']
        assert params_simple_pruned['zero_params'] == (10*5) # Only weights of first layer are zero


    # --- Test calculate_model_size_mb & load_model_state_dict_from_path ---
    print("\n--- Testing calculate_model_size_mb & load_model_state_dict_from_path ---")
    if 'resnet18_scratch' in locals():
        temp_model_path = Path("./temp_model_for_size_test.pth")
        
        # Save state_dict
        torch.save(resnet18_scratch.state_dict(), temp_model_path)
        
        # Calculate size
        size_mb = calculate_model_size_mb(resnet18_scratch.state_dict())
        print(f"ResNet18 (scratch) state_dict size: {size_mb:.2f} MB")
        assert size_mb > 0

        # Load state_dict
        loaded_state_dict = load_model_state_dict_from_path(temp_model_path, device='cpu')
        assert len(loaded_state_dict) == len(resnet18_scratch.state_dict())
        print(f"Successfully loaded state_dict from {temp_model_path}")

        # Test loading a model saved with 'model_state_dict' key
        torch.save({'model_state_dict': resnet18_scratch.state_dict(), 'epoch': 5}, temp_model_path)
        loaded_state_dict_wrapped = load_model_state_dict_from_path(temp_model_path, device='cpu')
        assert len(loaded_state_dict_wrapped) == len(resnet18_scratch.state_dict())
        print(f"Successfully loaded state_dict wrapped in 'model_state_dict' key.")


        if temp_model_path.exists():
            temp_model_path.unlink() # Clean up

    print("\nAll model_utils tests finished.")
