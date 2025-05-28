from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Any, Dict

@dataclass
class BaseConfig:
    """Base configuration for training and optimization runs."""
    # Paths
    data_dir: str = './data'  # Default updated to relative path
    save_dir: str = './saved_models_and_results'  # Default updated to relative path

    # Training parameters
    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    num_workers: int = 4

    # Model settings
    use_pretrained: bool = True
    evaluate_only: bool = False

    def __post_init__(self):
        """Validate configuration and convert paths."""
        if self.num_epochs < 0:
            raise ValueError("num_epochs must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate < 0:
            raise ValueError("learning_rate cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        # Convert to Path objects
        self.data_dir = Path(self.data_dir)
        self.save_dir = Path(self.save_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to dictionary with string paths."""
        data = asdict(self)
        # Convert Path objects back to strings for serialization
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data

@dataclass
class PruningConfig(BaseConfig):
    """Configuration for pruning experiments."""
    baseline_model_path: str = ""
    pruning_method: str = 'global_unstructured_l1'
    sparsity_rates: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    ft_epochs: int = 10
    ft_learning_rate: float = 0.0001
    resume_pruning: bool = False
    skip_completed: bool = True
    use_sparse_storage: bool = True
    continue_from_ft_model_path: Optional[str] = None
    ft_epochs_previous: int = 0
    ft_epochs_total: int = field(init=False) # Will be set in __post_init__

    def __post_init__(self):
        super().__post_init__()
        self.baseline_model_path = Path(self.baseline_model_path) if self.baseline_model_path else None
        if self.continue_from_ft_model_path:
            self.continue_from_ft_model_path = Path(self.continue_from_ft_model_path)
        
        # Initialize ft_epochs_total
        self.ft_epochs_total = self.ft_epochs

        # Validate sparsity rates
        if not all(0.0 <= rate < 1.0 for rate in self.sparsity_rates):
            raise ValueError("Sparsity rates must be between 0.0 and 1.0 (exclusive of 1.0).")
        if self.ft_epochs < 0:
            raise ValueError("ft_epochs must be non-negative.")
        if self.ft_learning_rate < 0:
            raise ValueError("ft_learning_rate cannot be negative.")
        if self.ft_epochs_previous < 0:
            raise ValueError("ft_epochs_previous cannot be negative.")

@dataclass
class QuantizationConfig(BaseConfig):
    """Configuration for quantization experiments."""
    baseline_model_path: str = ""
    quantization_method: str = 'ptq'  # 'ptq' or 'qat'
    calibration_batches: int = 100  # For PTQ
    qat_epochs: int = 5  # For QAT

    def __post_init__(self):
        super().__post_init__()
        self.baseline_model_path = Path(self.baseline_model_path) if self.baseline_model_path else None
        if self.calibration_batches < 0:
            raise ValueError("calibration_batches cannot be negative.")
        if self.qat_epochs < 0:
            raise ValueError("qat_epochs cannot be negative.")
        if self.quantization_method not in ['ptq', 'qat']:
            raise ValueError("quantization_method must be 'ptq' or 'qat'.")

@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for knowledge distillation experiments."""
    teacher_model_path: str = ""
    student_model_arch: str = 'resnet18'
    distillation_temperature: float = 2.0
    alpha_kd: float = 0.5  # Weight for KD loss (0 to 1)

    def __post_init__(self):
        super().__post_init__()
        self.teacher_model_path = Path(self.teacher_model_path) if self.teacher_model_path else None
        if not (0.0 <= self.alpha_kd <= 1.0):
            raise ValueError("alpha_kd (KD loss weight) must be between 0.0 and 1.0.")
        if self.distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive.")

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    base_cfg = BaseConfig()
    print("Base Config:", base_cfg.to_dict())

    pruning_cfg = PruningConfig(baseline_model_path="path/to/model.pth")
    print("Pruning Config:", pruning_cfg.to_dict())
    # Example of how ft_epochs_total would be used or updated
    # pruning_cfg.ft_epochs_total = pruning_cfg.ft_epochs_previous + pruning_cfg.ft_epochs 
    # print("Updated Pruning Config (ft_epochs_total):", pruning_cfg.to_dict())


    quant_cfg_ptq = QuantizationConfig(baseline_model_path="path/to/model.pth", quantization_method='ptq')
    print("Quantization PTQ Config:", quant_cfg_ptq.to_dict())

    quant_cfg_qat = QuantizationConfig(baseline_model_path="path/to/model.pth", quantization_method='qat', num_epochs=10, learning_rate=0.0005)
    print("Quantization QAT Config:", quant_cfg_qat.to_dict())

    distill_cfg = DistillationConfig(teacher_model_path="path/to/teacher.pth")
    print("Distillation Config:", distill_cfg.to_dict())

    # Test path conversion for save_dir
    assert isinstance(base_cfg.save_dir, Path), "save_dir should be a Path object"
    assert isinstance(pruning_cfg.save_dir, Path), "pruning_cfg.save_dir should be a Path object"
    if pruning_cfg.baseline_model_path:
      assert isinstance(pruning_cfg.baseline_model_path, Path), "pruning_cfg.baseline_model_path should be a Path object"

    # Test to_dict conversion
    pruning_cfg_dict = pruning_cfg.to_dict()
    assert isinstance(pruning_cfg_dict['save_dir'], str), "save_dir in dict should be a string"
    if pruning_cfg_dict['baseline_model_path']:
      assert isinstance(pruning_cfg_dict['baseline_model_path'], str), "baseline_model_path in dict should be a string"
    
    print("All example configs created and basic checks passed.")
