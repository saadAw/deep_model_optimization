#!/usr/bin/env python3
"""
Script to launch an iterative pruning experiment.
Configures and runs the IterativePruningExperiment class defined in pruning_script.py.
"""

import sys
from pathlib import Path
import time
from typing import List, Dict, Union


try:
    from pruning_script import PruningConfig, IterativePruningExperiment
except ImportError:
    print("Error: Could not import PruningConfig or IterativePruningExperiment from pruning_script.py.", file=sys.stderr)
    print("Please ensure 'pruning_script.py' is in the same directory or available in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

def main():
    """
    Main function to configure and run the iterative pruning experiment.
    """
    # --- CONFIGURE YOUR ITERATIVE PRUNING EXPERIMENT HERE ---

    # Define the iterative stages: Each dictionary specifies a cumulative target sparsity
    # and the number of fine-tuning epochs to run AFTER reaching that sparsity level.
    # The pruning amount will be calculated automatically by global_unstructured to reach
    # the 'target_sparsity' from the original model's total parameters.
    iterative_stages_config: List[Dict[str, Union[float, int]]] = [
        {'target_sparsity': 0.5, 'epochs': 1},   # Prune to 50% cumulative, then FT for 5 epochs
        {'target_sparsity': 0.75, 'epochs': 1},  # Prune to 75% cumulative, then FT for 10 epochs
        {'target_sparsity': 0.9, 'epochs': 1}    # Prune to 90% cumulative, then FT for 15 epochs
    ]

    # Calculate total epochs for the save directory name and for overall history tracking
    total_ft_epochs_across_stages = sum(stage['epochs'] for stage in iterative_stages_config)

    # Determine the final target sparsity for overall experiment file naming (e.g., metrics_90_iterative_l1.json)
    # This will be the target_sparsity of the last stage.
    final_overall_target_sparsity = iterative_stages_config[-1]['target_sparsity'] if iterative_stages_config else 0.0

    # Create a unique and descriptive save directory name for this specific iterative run
    # This helps keep your experiment results organized.
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    save_dir_name = (
        f"resnet50_iterative_l1_"
        f"{int(final_overall_target_sparsity*100)}pct_finalSP_" # Using final sparsity in name
        f"{len(iterative_stages_config)}stages_"
        f"{total_ft_epochs_across_stages}epochs_ft_"
        f"{timestamp}"
    )
    # The parent directory 'pruning_runs' will be created if it doesn't exist
    base_save_path = Path('./pruning_runs')
    specific_save_dir = base_save_path / save_dir_name
    specific_save_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

    # --- Initialize PruningConfig with Iterative Settings ---
    config = PruningConfig(
        # General model/dataset settings (important: ensure this path is correct!)
        baseline_model_path='./resnet50_baseline_e30_run/best_model.pth',
        save_dir=str(specific_save_dir), # Use the dynamically generated directory

        # Pruning strategy
        pruning_strategy_type='iterative_l1',
        # For iterative pruning, `sparsity_rates` will only contain the final overall target.
        # The individual stage targets are defined in `iterative_stages`.
        sparsity_rates=[final_overall_target_sparsity],

        # Iterative specific stages defined above
        iterative_stages=iterative_stages_config,

        # Fine-tuning parameters (these apply to each stage's fine-tuning phase)
        ft_learning_rate=0.00005,
        ft_momentum=0.9,
        ft_weight_decay=1e-4,
        # Note: `ft_epochs` from PruningConfig is mainly for one-shot.
        # For iterative, the epochs per stage are specified in `iterative_stages`.

        # Resume and skip settings
        resume_pruning=True, # Highly recommended for iterative runs to pick up if interrupted
        # `skip_completed=False` for iterative means it will try to run the *overall* experiment
        # even if a previous attempt partially completed, but it will pick up from the last
        # fine-tuning checkpoint *within* the overall iterative run (due to `resume_pruning=True`).
        # If set to True, it would skip the *entire* iterative run if `metrics_90_iterative_l1.json` exists.
        skip_completed=False,
        use_sparse_storage=True,
    )

    # --- Run the Experiment ---
    try:
        print(f"Starting iterative pruning experiment. Results will be saved in: {config.save_dir}")
        experiment = IterativePruningExperiment(config)
        experiment.run()
        print(f"Iterative pruning experiment finished. Check {config.save_dir} for results.")

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. Please check paths in configuration. Details: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration Error: Invalid value provided in the configuration. Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during the experiment: {e}", file=sys.stderr)
        # Print full traceback for debugging unexpected errors
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()