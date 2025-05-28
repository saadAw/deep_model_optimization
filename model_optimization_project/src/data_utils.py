import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path # For type hinting config.data_dir if needed, and robust path handling
import sys # For error printing

# Import BaseConfig from the config module within the same package
from .config import BaseConfig 
from typing import Tuple, Optional # For type hinting

# TODO: Future enhancement: Allow passing custom transform objects or selecting 
#       transform sets based on a config parameter (e.g., config.dataset_name).

def get_data_loaders(
    config: BaseConfig, 
    train_transforms_custom: Optional[transforms.Compose] = None,
    val_transforms_custom: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Sets up and returns data loaders for training and validation.

    Args:
        config: A BaseConfig object containing data_dir, batch_size, num_workers.
        train_transforms_custom: Optional custom transforms for the training set.
        val_transforms_custom: Optional custom transforms for the validation set.

    Returns:
        A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - num_classes (int): Number of classes in the dataset.
            
    Raises:
        FileNotFoundError: If the train or val directories are not found in config.data_dir.
    """

    # Default ImageNet normalization and transforms if custom ones are not provided
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train_transforms_custom is None:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transforms = train_transforms_custom

    if val_transforms_custom is None:
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        val_transforms = val_transforms_custom

    # Ensure data_dir is a Path object (BaseConfig should handle this in __post_init__)
    data_dir = Path(config.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    try:
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Training data directory not found: {train_dir}")
        if not val_dir.is_dir():
            raise FileNotFoundError(f"Validation data directory not found: {val_dir}")

        train_dataset = datasets.ImageFolder(str(train_dir), train_transforms)
        val_dataset = datasets.ImageFolder(str(val_dir), val_transforms)
    except FileNotFoundError as e:
        # Log detailed error message and re-raise
        print(f"ERROR: Dataset directory not found. {e}", file=sys.stderr)
        print(f"Ensure {config.data_dir} contains 'train' and 'val' subdirectories.", file=sys.stderr)
        raise
    except Exception as e: # Catch other potential datasets.ImageFolder errors
        print(f"ERROR: Failed to load dataset from {config.data_dir}: {e}", file=sys.stderr)
        raise


    num_classes = len(train_dataset.classes)
    if num_classes == 0:
        # This case should ideally be caught by ImageFolder error if train_dir is empty or malformed
        raise ValueError(f"No classes found in training dataset at {train_dir}. Ensure it's populated correctly.")


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0  # Avoid worker recreation overhead
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )

    return train_loader, val_loader, num_classes

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Create dummy data directories and dummy image files for testing
    # Note: This requires Pillow to be installed for ImageFolder to not error on dummy images.
    # In a real CI/testing environment, you might need to install `Pillow`.
    # For this example, we'll assume the directories can be created and ImageFolder might
    # still work or fail gracefully if no actual images are needed for the test logic here.

    print("Running example usage of get_data_loaders...")
    
    # Create a dummy BaseConfig
    dummy_config = BaseConfig(
        data_dir='./temp_data_for_test', 
        batch_size=2, 
        num_workers=0 # Simpler for local test without multiprocessing issues
    )

    temp_data_path = Path(dummy_config.data_dir)
    temp_train_path = temp_data_path / 'train'
    temp_val_path = temp_data_path / 'val'

    # Create dummy class folders and a dummy file to make ImageFolder work
    # (ImageFolder needs at least one file in one subfolder)
    (temp_train_path / 'class_a').mkdir(parents=True, exist_ok=True)
    (temp_train_path / 'class_b').mkdir(parents=True, exist_ok=True)
    (temp_val_path / 'class_a').mkdir(parents=True, exist_ok=True)
    (temp_val_path / 'class_b').mkdir(parents=True, exist_ok=True)

    try:
        # Create dummy files (actual image files are not strictly necessary for this basic test)
        # Pillow would be needed if datasets.ImageFolder tries to open them.
        # For this test, we just need the directory structure.
        (temp_train_path / 'class_a' / 'dummy.txt').touch() 
        (temp_train_path / 'class_b' / 'dummy.txt').touch()
        (temp_val_path / 'class_a' / 'dummy.txt').touch()
        (temp_val_path / 'class_b' / 'dummy.txt').touch()

        print(f"Attempting to load data from temporary directory: {temp_data_path.resolve()}")

        train_loader, val_loader, num_classes = get_data_loaders(dummy_config)

        print(f"Successfully created data loaders.")
        print(f"Number of classes: {num_classes}")
        assert num_classes == 2, f"Expected 2 classes, got {num_classes}"
        
        print(f"Train loader batch size: {train_loader.batch_size}")
        assert train_loader.batch_size == dummy_config.batch_size
        
        print(f"Val loader batch size: {val_loader.batch_size}")
        assert val_loader.batch_size == dummy_config.batch_size

        # Basic check if loaders can iterate (optional, requires actual data or mock)
        # try:
        #     for i, (data, target) in enumerate(train_loader):
        #         print(f"Train batch {i+1}: data shape {data.shape}, target shape {target.shape}")
        #         if i >= 1: # Check only a couple of batches
        #             break
        # except Exception as e:
        #     print(f"Could not iterate through train_loader (requires Pillow and valid images): {e}")

        print("Data loader tests passed (structure and basic parameters).")

    except FileNotFoundError as e:
        print(f"Test failed: {e}")
        print("Please ensure the script has permissions to create directories or that the dummy paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")
        print("This might be due to missing dependencies like Pillow for ImageFolder, or issues with dummy data creation.")
    finally:
        # Clean up dummy directories
        import shutil
        if temp_data_path.exists():
            print(f"Cleaning up temporary data directory: {temp_data_path}")
            shutil.rmtree(temp_data_path)
        print("Example usage finished.")
