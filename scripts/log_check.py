import os
import json
from collections import Counter, defaultdict

def find_log_files(root_dir, filename="log.json"):
    """Recursively finds all files with a specific name in a directory."""
    log_file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            log_file_paths.append(os.path.join(dirpath, filename))
    return log_file_paths

def get_all_keys_from_dict(data, prefix=""):
    """
    Recursively extracts all keys from a nested dictionary,
    representing nested keys with dot notation.
    """
    keys = set()
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{prefix}{k}"
            keys.add(full_key)
            keys.update(get_all_keys_from_dict(v, f"{full_key}."))
    elif isinstance(data, list):
        # If you want to analyze structure within lists (e.g., keys in dicts inside a list)
        # you could extend this, but for now, we'll just mark the list key itself.
        # For example, if 'epoch_data' is a list of dicts, we'd get 'epoch_data' as a key.
        # If you need to know if 'epoch_data.epoch' exists, that's more complex.
        # Let's assume for now we're interested in the keys of the main dictionary structure.
        pass # Or handle list items if they are dicts:
        # for i, item in enumerate(data):
        #     keys.update(get_all_keys_from_dict(item, f"{prefix}[{i}]."))
    return keys

def analyze_log_keys(root_directory="saved_models_and_logs"):
    """
    Analyzes all log.json files to find common and missing keys.
    """
    log_files = find_log_files(root_directory)
    if not log_files:
        print(f"No 'log.json' files found in {root_directory}")
        return

    print(f"Found {len(log_files)} 'log.json' files to analyze.\n")

    all_keys_counter = Counter() # Counts how many files each key appears in
    file_specific_keys = defaultdict(set) # Stores the set of keys for each file
    example_log_for_key = {} # Store one example log path for each key

    for log_path in log_files:
        try:
            with open(log_path, 'r') as f:
                content = json.load(f)
            
            # Check for 'epoch_data' specifically and its emptiness
            if 'epoch_data' in content:
                if not content['epoch_data']: # Empty list
                    file_specific_keys[log_path].add("epoch_data (empty_list)")
                else:
                    file_specific_keys[log_path].add("epoch_data (non-empty_list)")
            
            keys_in_file = get_all_keys_from_dict(content)
            file_specific_keys[log_path].update(keys_in_file)

            for key in keys_in_file:
                all_keys_counter[key] += 1
                if key not in example_log_for_key:
                    example_log_for_key[key] = log_path

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {log_path}")
        except Exception as e:
            print(f"An error occurred while processing {log_path}: {e}")

    print("\n--- Key Presence Analysis ---")
    num_total_files = len(log_files)

    print(f"\nKeys present in ALL {num_total_files} log files:")
    universally_present = True
    for key, count in all_keys_counter.items():
        if count == num_total_files:
            print(f"  - {key}")
            universally_present = False
    if universally_present and num_total_files > 0: # Check if any key was actually universal
        pass # Handled by the loop
    elif num_total_files == 0:
        pass
    elif universally_present: # means no keys were printed in the loop
         print("  (None)")


    print(f"\nKeys present in SOME but NOT ALL log files (out of {num_total_files} files):")
    partially_present = False
    sorted_partial_keys = sorted(
        [(key, count) for key, count in all_keys_counter.items() if count < num_total_files],
        key=lambda item: item[1],
        reverse=True
    )
    for key, count in sorted_partial_keys:
        print(f"  - {key} (present in {count} files)")
        partially_present = True
    if not partially_present and num_total_files > 0 :
        print("  (None)")


    # Detailed breakdown: which files are missing common keys?
    # Let's define "common" as present in > 50% of files but not all
    print(f"\n--- Detailed Breakdown for Partially Present Keys ---")
    common_threshold = num_total_files * 0.5
    at_least_one_common_missing_reported = False

    for key, count in sorted_partial_keys:
        if count > common_threshold or key.startswith("original_evaluation_metrics_from_log") or key.startswith("training_summary.achieved_overall_parameter_sparsity_percent") or key.startswith("training_summary.accuracy_before_ft"): # Focus on relatively common ones or known conditional ones
            missing_in_files = []
            for log_path, keys_in_this_file_set in file_specific_keys.items():
                # Need to check against the original keys extracted by get_all_keys_from_dict
                # The file_specific_keys might have the "epoch_data (empty_list)" variant
                original_keys_for_file = get_all_keys_from_dict(json.load(open(log_path, 'r')))
                if key not in original_keys_for_file:
                    missing_in_files.append(os.path.relpath(log_path, root_directory))
            
            if missing_in_files:
                print(f"\nKey '{key}' (present in {count}/{num_total_files} files) is MISSING in:")
                for rel_path in sorted(missing_in_files[:5]): # Print first 5 missing
                    print(f"  - {rel_path}")
                if len(missing_in_files) > 5:
                    print(f"  ... and {len(missing_in_files) - 5} more files.")
                at_least_one_common_missing_reported = True
                
    if not at_least_one_common_missing_reported and any(count > common_threshold for _, count in sorted_partial_keys):
         print("  (No commonly present keys found to be missing in specific files, or all partial keys are very rare)")
    elif not sorted_partial_keys:
         print("  (No partially present keys to analyze for missing instances)")


    # Special check for epoch_data variants
    print("\n--- 'epoch_data' Variants Analysis ---")
    epoch_data_empty_count = 0
    epoch_data_non_empty_count = 0
    epoch_data_absent_count = 0

    for log_path in log_files:
        keys_in_this_file = file_specific_keys[log_path]
        if "epoch_data (empty_list)" in keys_in_this_file:
            epoch_data_empty_count += 1
        elif "epoch_data (non-empty_list)" in keys_in_this_file:
            epoch_data_non_empty_count += 1
        else: # 'epoch_data' key itself was not found by get_all_keys_from_dict
            with open(log_path, 'r') as f:
                content = json.load(f)
            if 'epoch_data' not in content:
                 epoch_data_absent_count +=1


    print(f"  - 'epoch_data' key present with an EMPTY list: {epoch_data_empty_count} files")
    print(f"  - 'epoch_data' key present with a NON-EMPTY list: {epoch_data_non_empty_count} files")
    print(f"  - 'epoch_data' key ABSENT entirely: {epoch_data_absent_count} files")


if __name__ == "__main__":
    # Make sure this script is in the parent directory of 'saved_models_and_logs'
    # or provide the correct path.
    project_root = "." # Assumes script is in the same dir as 'saved_models_and_logs'
    target_folder = os.path.join(project_root, "saved_models_and_logs") 
    
    # If your script is one level up from saved_models_and_logs:
    # target_folder = "saved_models_and_logs"

    if not os.path.isdir(target_folder):
        print(f"Error: Directory '{target_folder}' not found. Please check the path.")
    else:
        analyze_log_keys(target_folder)