import torch
import torch.nn as nn
import torch.ao.quantization as ao_quant # For modern quantization API
from torch.utils.data import DataLoader
from typing import Optional, Any, List, Type, Tuple # Added List, Type, Tuple
import copy

# Project-specific imports (assuming they are in the same src directory)
# from .model_utils import build_model # Example, if needed for creating quantizable models
# from .logger_utils import TrainingLogger # Example, if specific logging is needed


def fuse_model_modules(
    model: nn.Module, 
    module_list_to_fuse: Optional[List[List[str]]] = None,
    is_qat: bool = False,
    logger: Optional[Any] = None
) -> nn.Module:
    """
    Fuses specified sequences of modules in a model. This is typically done
    before quantization (both PTQ and QAT).

    Args:
        model (nn.Module): The model to fuse.
        module_list_to_fuse (Optional[List[List[str]]]): A list of module name sequences
            to fuse. For example, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']].
            If None, common ResNet/MobileNet fusions might be attempted if identifiable,
            but it's best to specify. For torchvision's quantizable models,
            they often have a .fuse_model() method.
        is_qat (bool): Indicates if fusion is for QAT. This might affect fusion behavior
                       or which `fuse_model` method is called if the model has one.
        logger (Optional[Any]): An optional logger.

    Returns:
        nn.Module: The model with fused modules (modified in-place).
    """
    model.eval() # Fusion is typically done in eval mode, QAT will switch back to train.
    
    if hasattr(model, 'fuse_model'):
        if logger:
            logger.info("Model has a built-in 'fuse_model' method. Calling it.")
        # torchvision.models.quantization.<Arch> often have a fuse_model(is_qat=...) method
        try:
            if 'is_qat' in model.fuse_model.__code__.co_varnames: # Check if is_qat is an arg
                 model.fuse_model(is_qat=is_qat)
            else:
                 model.fuse_model() # Call without is_qat if not supported by the method
            if logger:
                logger.info("Built-in 'fuse_model' called successfully.")
            return model
        except Exception as e:
            if logger:
                logger.error(f"Error calling model.fuse_model(): {e}. Attempting manual fusion if module_list_to_fuse is provided.")
    
    if module_list_to_fuse is None or not module_list_to_fuse:
        if logger:
            logger.warning("module_list_to_fuse not provided and model has no generic 'fuse_model' method. No fusion performed.")
        return model

    if logger:
        logger.info(f"Attempting manual fusion for {len(module_list_to_fuse)} module sequences.")

    try:
        model = ao_quant.fuse_modules(model, module_list_to_fuse, inplace=True)
        if logger:
            logger.info(f"Successfully fused modules for sequences: {module_list_to_fuse}")
    except Exception as e:
        if logger:
            logger.error(f"Error during ao_quant.fuse_modules for sequences {module_list_to_fuse}: {e}")
            # Log which specific sequence might have failed if possible (requires more granular try-except)
            
    return model


def prepare_ptq_static_model(
    model: nn.Module, 
    qconfig_backend: str = 'fbgemm', # 'fbgemm' for x86, 'qnnpack' for ARM
    per_channel_quant: bool = False, # Whether to use per-channel quantization for weights
    logger: Optional[Any] = None
) -> nn.Module:
    """
    Prepares a model for Post-Training Static Quantization.
    This involves:
    1. Ensuring the model is in evaluation mode.
    2. Assigning a quantization configuration (qconfig).
    3. Calling `torch.ao.quantization.prepare` to insert observers.

    Args:
        model (nn.Module): The model to prepare (should be fused already if applicable).
        qconfig_backend (str): Backend for qconfig ('fbgemm' or 'qnnpack').
        per_channel_quant (bool): If True, use per-channel quantization for weights.
                                 Otherwise, use per-tensor.
        logger (Optional[Any]): An optional logger.

    Returns:
        nn.Module: The model prepared for calibration (observers inserted).
    """
    model.eval() # PTQ preparation and calibration require eval mode.

    if logger:
        logger.info(f"Preparing model for PTQ Static with backend: {qconfig_backend}, per-channel: {per_channel_quant}")

    if qconfig_backend not in ['fbgemm', 'qnnpack']:
        if logger:
            logger.warning(f"Unsupported qconfig_backend '{qconfig_backend}'. Defaulting to 'fbgemm'.")
        qconfig_backend = 'fbgemm'
    
    # Select appropriate qconfig
    if per_channel_quant:
        # Example of a common per-channel qconfig.
        # For fbgemm: symmetric quantization for weights, affine for activations.
        # For qnnpack: affine for weights, affine for activations.
        # These details can be customized further.
        if qconfig_backend == 'fbgemm':
            qconfig = ao_quant.QConfig(
                activation=ao_quant.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                weight=ao_quant.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
            )
        else: # qnnpack
             qconfig = ao_quant.QConfig(
                activation=ao_quant.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True),
                weight=ao_quant.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine, reduce_range=False, ch_axis=0)
            )
    else: # Per-tensor quantization
        qconfig = ao_quant.get_default_qconfig(qconfig_backend)

    model.qconfig = qconfig
    if logger:
        logger.info(f"Applied qconfig: {model.qconfig}")

    try:
        # `prepare` inserts observers in place.
        # For models with QuantStub/DeQuantStub, this is straightforward.
        # For models without, you might need to manually add them or use a quantizable model version.
        # This function assumes the model is structured to be compatible with `prepare`.
        ao_quant.prepare(model, inplace=True) 
        if logger:
            logger.info("ao_quant.prepare called successfully. Observers inserted.")
    except Exception as e:
        if logger:
            logger.error(f"Error during ao_quant.prepare: {e}. Model might need QuantStub/DeQuantStub or be a quantizable variant.")
        raise # Re-raise the exception as this is a critical step.
        
    return model


def calibrate_ptq_model(
    model_prepared: nn.Module, 
    data_loader: DataLoader, 
    device: torch.device, 
    num_calibration_batches: int = 100,
    logger: Optional[Any] = None
):
    """
    Calibrates a model prepared for PTQ static quantization by feeding it data.

    Args:
        model_prepared (nn.Module): Model with observers inserted (from prepare_ptq_static_model).
        data_loader (DataLoader): DataLoader for calibration data.
        device (torch.device): Device to perform calibration on.
        num_calibration_batches (int): Number of batches from data_loader to use for calibration.
        logger (Optional[Any]): An optional logger.
    """
    model_prepared.eval() # Ensure model is in eval mode for calibration
    model_prepared.to(device)

    actual_batches_to_use = min(num_calibration_batches, len(data_loader))

    if logger:
        logger.info(f"Starting PTQ calibration on device '{device}' with {actual_batches_to_use} batches.")
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= actual_batches_to_use:
                break
            inputs = inputs.to(device)
            model_prepared(inputs) # Forward pass to collect statistics
            
            if logger and (batch_idx + 1) % 20 == 0: # Log progress every 20 batches
                logger.info(f"Calibration progress: Batch {batch_idx + 1}/{actual_batches_to_use}")
    
    if logger:
        logger.info(f"PTQ calibration finished after {actual_batches_to_use} batches.")


def convert_ptq_static_model(
    model_calibrated: nn.Module, 
    logger: Optional[Any] = None
) -> nn.Module:
    """
    Converts a calibrated model to a quantized model for PTQ static quantization.

    Args:
        model_calibrated (nn.Module): Calibrated model (observers have collected statistics).
        logger (Optional[Any]): An optional logger.

    Returns:
        nn.Module: The quantized model.
    """
    model_calibrated.eval() # Ensure model is in eval mode for conversion
    # Conversion typically happens on CPU if the target is CPU deployment
    # model_calibrated.cpu() # Optional: move to CPU before conversion if not already

    if logger:
        logger.info("Converting calibrated model to quantized version...")
    
    try:
        # `convert` replaces modules with their quantized counterparts in place.
        ao_quant.convert(model_calibrated, inplace=True)
        if logger:
            logger.info("ao_quant.convert called successfully. Model is now quantized.")
    except Exception as e:
        if logger:
            logger.error(f"Error during ao_quant.convert: {e}")
        raise # Re-raise as this is critical
        
    return model_calibrated # Return the same model, modified in-place


def prepare_qat_model(
    model: nn.Module, 
    qconfig_backend: str = 'fbgemm', # 'fbgemm' for x86, 'qnnpack' for ARM
    logger: Optional[Any] = None
) -> nn.Module:
    """
    Prepares a model for Quantization-Aware Training (QAT).
    This involves:
    1. Ensuring the model is in training mode (or eval if only fusing, then train for QAT).
    2. Assigning a QAT-specific quantization configuration.
    3. Calling `torch.ao.quantization.prepare_qat` to insert fake quantization modules.

    Args:
        model (nn.Module): The model to prepare (should be fused already if applicable).
        qconfig_backend (str): Backend for qconfig ('fbgemm' or 'qnnpack').
        logger (Optional[Any]): An optional logger.

    Returns:
        nn.Module: The model prepared for QAT.
    """
    # QAT preparation starts typically with the model in eval mode for fusion,
    # then switched to train mode for the prepare_qat step and subsequent training.
    # If fusion was done separately:
    model.train() # QAT requires model to be in train mode for `prepare_qat`

    if logger:
        logger.info(f"Preparing model for QAT with backend: {qconfig_backend}")

    if qconfig_backend not in ['fbgemm', 'qnnpack']:
        if logger:
            logger.warning(f"Unsupported qconfig_backend '{qconfig_backend}'. Defaulting to 'fbgemm'.")
        qconfig_backend = 'fbgemm'
        
    model.qconfig = ao_quant.get_default_qat_qconfig(qconfig_backend)
    if logger:
        logger.info(f"Applied QAT qconfig: {model.qconfig}")

    try:
        # `prepare_qat` inserts fake quantization modules in place.
        # Model should be fused before this step if fusion is desired.
        ao_quant.prepare_qat(model, inplace=True)
        if logger:
            logger.info("ao_quant.prepare_qat called successfully. Fake quantization modules inserted.")
    except Exception as e:
        if logger:
            logger.error(f"Error during ao_quant.prepare_qat: {e}")
        raise # Re-raise
        
    return model


# --- Example Usage ---
if __name__ == '__main__':
    from torchvision.models.quantization import resnet18 as q_resnet18 # Quantizable ResNet18
    from .logger_utils import TrainingLogger

    test_save_dir = Path("./temp_quant_utils_test_run")
    test_save_dir.mkdir(parents=True, exist_ok=True)
    
    test_logger = TrainingLogger(save_dir=test_save_dir, log_file_name="quant_utils_test.log")
    test_logger.info("--- Starting quantization_utils.py tests ---")

    device = torch.device("cpu") # Quantization is often targeted for CPU
    test_logger.info(f"Using device: {device}")

    # 1. Create a quantizable model instance
    # Using torchvision's quantizable models which already have QuantStub/DeQuantStub
    fp32_model = q_resnet18(weights=None, quantize=False) # Get FP32 version first
    fp32_model.fc = nn.Linear(fp32_model.fc.in_features, 10) # Adjust classifier
    fp32_model.to(device)
    fp32_model.eval() # Start in eval mode

    # Create a copy for QAT tests
    qat_fp32_model_orig = copy.deepcopy(fp32_model)

    test_logger.info("Created quantizable ResNet18 model for testing.")

    # Dummy DataLoader for calibration
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=20, img_size=(3, 224, 224)):
            self.num_samples = num_samples
            self.data = torch.randn(num_samples, *img_size)
            self.targets = torch.randint(0, 10, (num_samples,)) # Dummy targets for 10 classes
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    calib_dataset = DummyDataset(num_samples=32) # Small dataset for quick test
    calib_loader = DataLoader(calib_dataset, batch_size=8)
    test_logger.info(f"Created dummy calibration DataLoader with {len(calib_loader)} batches.")

    # --- Test PTQ Static Workflow ---
    test_logger.info("\n--- Testing PTQ Static Workflow ---")
    try:
        # a. Fuse model (quantizable ResNet18 has its own fuse_model method)
        ptq_model_fused = fuse_model_modules(fp32_model, is_qat=False, logger=test_logger)
        
        # b. Prepare for PTQ
        # Using per-tensor for simplicity in example, though per-channel is often better for accuracy
        ptq_model_prepared = prepare_ptq_static_model(ptq_model_fused, qconfig_backend='fbgemm', per_channel_quant=False, logger=test_logger)
        
        # c. Calibrate
        calibrate_ptq_model(ptq_model_prepared, calib_loader, device, num_calibration_batches=2, logger=test_logger)
        
        # d. Convert to quantized model
        quantized_ptq_model = convert_ptq_static_model(ptq_model_prepared, logger=test_logger)
        test_logger.info("PTQ Static workflow completed successfully.")
        
        # Basic check: run inference with the quantized model
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        quantized_ptq_model.eval() # Ensure it's in eval mode after conversion
        with torch.no_grad():
            output = quantized_ptq_model(dummy_input)
        test_logger.info(f"PTQ model output shape: {output.shape}")
        assert output.shape[0] == 1 and output.shape[1] == 10 # Batch size 1, 10 classes
        
        # Check if some layers are quantized (e.g., conv1)
        assert isinstance(quantized_ptq_model.conv1, ao_quant.QuantizedConv2d), \
            f"Expected conv1 to be QuantizedConv2d, got {type(quantized_ptq_model.conv1)}"
        test_logger.info("Verified that conv1 layer is quantized in PTQ model.")

    except Exception as e:
        test_logger.error(f"Error during PTQ Static workflow test: {e}", exc_info=True)


    # --- Test QAT Workflow (Preparation part) ---
    test_logger.info("\n--- Testing QAT Preparation Workflow ---")
    try:
        # a. Fuse model for QAT
        qat_model_fused = fuse_model_modules(qat_fp32_model_orig, is_qat=True, logger=test_logger)
        
        # b. Prepare for QAT
        qat_model_prepared = prepare_qat_model(qat_model_fused, qconfig_backend='fbgemm', logger=test_logger)
        test_logger.info("QAT preparation completed successfully.")
        
        # Basic check: model should be in training mode after prepare_qat
        assert qat_model_prepared.training, "Model should be in training mode after QAT preparation."
        
        # Check if some layers have fake quantization (e.g., conv1)
        # The actual module type might be the original (e.g., nn.Conv2d) but it will have
        # 'activation_post_process' (observer/fake_quant) and 'weight_fake_quant' attributes.
        assert hasattr(qat_model_prepared.conv1, 'weight_fake_quant'), \
            "Expected conv1 to have 'weight_fake_quant' after QAT prepare."
        test_logger.info("Verified that conv1 layer has QAT attributes.")

        # Dummy forward pass (simulating start of training)
        qat_model_prepared.train() # Ensure it's in train mode for QAT
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = qat_model_prepared(dummy_input) # No torch.no_grad() during QAT training
        test_logger.info(f"QAT-prepared model output shape: {output.shape}")
        assert output.shape[0] == 1 and output.shape[1] == 10

    except Exception as e:
        test_logger.error(f"Error during QAT preparation workflow test: {e}", exc_info=True)

    test_logger.info("\nAll quantization_utils tests finished.")
    print(f"\nExample finished. Check logs in {test_save_dir.resolve()}")

    # Cleanup
    # import shutil
    # shutil.rmtree(test_save_dir)
    # print(f"Cleaned up directory: {test_save_dir.resolve()}")
