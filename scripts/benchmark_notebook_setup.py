# This is a helper file to make the notebook cleaner. It will contain our functions.
import os
import sys
import subprocess
import json
import shutil
import torch
import torchvision
import pandas as pd

def check_prerequisites():
    """Checks if trtexec is installed and available in the system's PATH."""
    print("Checking for 'trtexec'...")
    if shutil.which("trtexec") is None:
        print("❌ ERROR: 'trtexec' command not found.")
        raise RuntimeError("trtexec not found. Cannot proceed with benchmarking.")
    print("✅ 'trtexec' is available.")

def export_to_onnx(model_path, onnx_path, num_classes, batch_size, input_name, input_shape):
    """Loads a PyTorch model and exports it to ONNX format."""
    # ... (This function is correct, no changes needed) ...
    print(f"  Exporting '{os.path.basename(model_path)}' to ONNX...")
    if not os.path.exists(model_path):
        print(f"  ❌ ERROR: Model file not found at {model_path}")
        return False
    try:
        shape_parts = [int(d) for d in input_shape.split('x')]
        dummy_input_shape = (batch_size, *shape_parts)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)
        dummy_input = torch.randn(dummy_input_shape, device=device)
        torch.onnx.export(
            model, dummy_input, onnx_path, export_params=True, opset_version=13,
            do_constant_folding=True, input_names=[input_name], output_names=['output'],
            dynamic_axes={input_name: {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"  ✅ Successfully exported to {onnx_path}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED to export to ONNX: {e}")
        return False

def run_benchmark(onnx_path, engine_path, json_path, log_path, config, is_sparse):
    """Constructs and runs the trtexec benchmark command."""
    # ... (This function is correct from the last fix, no changes needed) ...
    print(f"  Building and benchmarking with trtexec...")
    opt_shape_str = f"{config['INPUT_NAME']}:{config['BATCH_SIZE']}x{config['INPUT_SHAPE']}"
    command = [
        "trtexec", f"--onnx={onnx_path}", "--fp16", "--useCudaGraph",
        f"--optShapes={opt_shape_str}", f"--iterations={config['ITERATIONS']}",
        f"--duration={config['DURATION']}", f"--saveEngine={engine_path}",
        f"--exportTimes={json_path}",
    ]
    if is_sparse:
        command.append("--sparsity=enable")
    command_str = " ".join(command) + f" > {log_path} 2>&1"
    print(f"  Running command: {command_str}")
    try:
        subprocess.run(command_str, shell=True, check=True)
        print(f"  ✅ trtexec benchmark completed. Log saved to {log_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ trtexec FAILED.")
        print(f"  Return code: {e.returncode}")
        print(f"  Check the full log file for details: {log_path}")
        return False

def parse_results(json_path, batch_size):
    """Parses the JSON output from trtexec to get key performance metrics."""
    # ===================================================================
    # --- THIS IS THE CORRECTED FUNCTION ---
    # ===================================================================
    try:
        with open(json_path, 'r') as f:
            # The JSON file might contain multiple JSON objects. We only need the last one.
            # Read all lines and find the last valid JSON object.
            json_text = ""
            for line in f:
                # A simple heuristic: the results object starts with '{'
                if line.strip().startswith('{'):
                    json_text = line
            
            # If no line started with '{', something is wrong.
            if not json_text:
                print(f"  ⚠️ Warning: Could not find a valid JSON object in {json_path}")
                return None
            
            data = json.loads(json_text)

        # Now, safely access the keys
        results_dict = data.get("results", {})
        
        throughput_qps = results_dict.get("throughput(qps)", 0)
        images_per_second = throughput_qps * batch_size

        latency_ms_list = results_dict.get("latency", [0])
        median_latency_ms = sorted(latency_ms_list)[len(latency_ms_list) // 2]

        gpu_compute_ms_list = results_dict.get("GPU-compute", [0])
        median_gpu_compute_ms = sorted(gpu_compute_ms_list)[len(gpu_compute_ms_list) // 2]
        
        return {
            "Throughput (images/sec)": images_per_second,
            "Batch Latency (ms)": median_latency_ms,
            "GPU Compute Time (ms)": median_gpu_compute_ms
        }
    except Exception as e:
        print(f"  ❌ Error parsing {json_path}: {e}")
        return None
