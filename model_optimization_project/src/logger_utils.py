import logging
import sys
import json
from pathlib import Path
from typing import Optional # Added for Optional type hint

# Import BaseConfig from the config module within the same package
from .config import BaseConfig


class TrainingLogger:
    """Handles logging setup and configuration logging."""

    def __init__(self, save_dir: Path, log_file_name: str = 'run.log', logger_name: Optional[str] = None):
        self.save_dir = Path(save_dir) # Ensure save_dir is a Path object
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.save_dir / log_file_name

        # Setup logger with a unique or specified name to avoid conflicts
        # If no logger_name is provided, use a default based on object id
        _logger_name = logger_name if logger_name else f"training_logger_{id(self)}"
        self.logger = logging.getLogger(_logger_name)
        
        # Only configure if not already configured (e.g., if logger was retrieved by name and already has handlers)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # File handler
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout) # Log to stdout
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log_config(self, config: BaseConfig):
        """Log and save run configuration using BaseConfig."""
        self.logger.info("=== Configuration ===")
        # Use the to_dict() method from BaseConfig (and its subclasses)
        config_dict = config.to_dict() 
        for key, value in config_dict.items():
            self.logger.info(f"{key}: {value}")

        # Save config to file
        config_path = self.save_dir / 'config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.logger.info(f"Config saved to {config_path}")
        except IOError as e:
            self.logger.error(f"Failed to save config file: {e}")

    def log_dataset_info(self, train_size: int, val_size: int, num_classes: int):
        """Log dataset information."""
        self.logger.info("=== Dataset Information ===")
        self.logger.info(f"Number of classes: {num_classes}")
        self.logger.info(f"Training samples: {train_size:,}") # Use comma for thousands separator
        self.logger.info(f"Validation samples: {val_size:,}") # Use comma for thousands separator

    def info(self, message: str):
        """Log an informational message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log a critical message."""
        self.logger.critical(message)

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Create a dummy config for testing
    class DummyConfig(BaseConfig):
        extra_param: str = "test_value"

    # Test with BaseConfig
    base_cfg = BaseConfig(save_dir='./temp_logs_base', num_epochs=5)
    base_logger = TrainingLogger(save_dir=base_cfg.save_dir, log_file_name='base_run.log', logger_name="BaseLoggerTest")
    base_logger.log_config(base_cfg)
    base_logger.log_dataset_info(train_size=10000, val_size=2000, num_classes=10)
    base_logger.info("This is an info message from base_logger.")
    
    # Test with a derived config (DummyConfig)
    dummy_cfg = DummyConfig(save_dir='./temp_logs_dummy', num_epochs=3, extra_param="another_value")
    dummy_logger = TrainingLogger(save_dir=dummy_cfg.save_dir, log_file_name='dummy_run.log', logger_name="DummyLoggerTest")
    dummy_logger.log_config(dummy_cfg) # Should use DummyConfig's to_dict() via BaseConfig
    dummy_logger.info(f"Dummy param: {dummy_cfg.extra_param}")

    print(f"Base logs should be in {base_cfg.save_dir}")
    print(f"Dummy logs should be in {dummy_cfg.save_dir}")

    # Check if config.json was created and if it contains path as string
    expected_base_config_path = base_cfg.save_dir / 'config.json'
    assert expected_base_config_path.exists(), f"Config file not found at {expected_base_config_path}"
    with open(expected_base_config_path, 'r') as f:
        base_config_data = json.load(f)
    assert isinstance(base_config_data['save_dir'], str), "save_dir in base_config.json should be a string"
    assert base_config_data['extra_param'] is None, "extra_param should not be in base_config.json"


    expected_dummy_config_path = dummy_cfg.save_dir / 'config.json'
    assert expected_dummy_config_path.exists(), f"Config file not found at {expected_dummy_config_path}"
    with open(expected_dummy_config_path, 'r') as f:
        dummy_config_data = json.load(f)
    assert isinstance(dummy_config_data['save_dir'], str), "save_dir in dummy_config.json should be a string"
    assert dummy_config_data['extra_param'] == "another_value", "extra_param not correctly saved in dummy_config.json"

    print("Logger example usage finished. Check console and log files.")
    print("To clean up, remove ./temp_logs_base and ./temp_logs_dummy directories.")

    # Test that loggers with the same name and path don't duplicate handlers
    logger_shared = TrainingLogger(save_dir=Path("./temp_logs_base"), logger_name="BaseLoggerTest")
    assert len(logger_shared.logger.handlers) == 2, "Logger should reuse existing handlers if name is the same."
    
    # Test that different logger names result in different logger instances with their own handlers
    logger_new_name = TrainingLogger(save_dir=Path("./temp_logs_base"), logger_name="NewLoggerTest")
    assert len(logger_new_name.logger.handlers) == 2, "New logger should have its own handlers."
    assert logger_new_name.logger is not logger_shared.logger, "Loggers with different names should be different instances."
    
    print("Logger uniqueness tests passed.")
