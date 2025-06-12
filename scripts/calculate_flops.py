import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import glob
import gc
import warnings
warnings.filterwarnings('ignore')

# Try multiple profiling libraries for robustness
profiling_tools = {}
try:
    from thop import profile, clever_format
    profiling_tools['thop'] = True
except ImportError:
    profiling_tools['thop'] = False

try:
    from torchinfo import summary
    profiling_tools['torchinfo'] = True
except ImportError:
    profiling_tools['torchinfo'] = False

try:
    from ptflops import get_model_complexity_info
    profiling_tools['ptflops'] = True
except ImportError:
    profiling_tools['ptflops'] = False

print("Available profiling tools:", {k: v for k, v in profiling_tools.items() if v})

# --- Configuration ---
MODELS_ROOT_DIR = "saved_models_and_logs"
OUTPUT_CSV = "flops.csv"
DEVICE = torch.device("cpu")
INPUT_SIZE = (3, 224, 224)
BATCH_SIZE = 1

def get_base_model(arch_name="resnet50", num_classes=1000):
    """Create base model architecture"""
    arch_name_low = arch_name.lower()
    if arch_name_low == "resnet50":
        return models.resnet50(weights=None, num_classes=num_classes)
    elif arch_name_low == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)
    else:
        print(f"Warning: Unknown architecture '{arch_name}', defaulting to ResNet50")
        return models.resnet50(weights=None, num_classes=num_classes)

def extract_model_info(model):
    """Extract basic model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers by type
    layer_counts = {}
    for module in model.modules():
        module_type = type(module).__name__
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_counts': layer_counts
    }

def calculate_flops_thop(model, input_size=(1, 3, 224, 224)):
    """Calculate FLOPs using thop"""
    try:
        model.eval()
        dummy_input = torch.randn(input_size)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return {
            'flops_gmacs': macs / 1e9,
            'params_millions': params / 1e6,
            'method': 'thop'
        }
    except Exception as e:
        return {'error': f"thop failed: {str(e)[:100]}"}

def calculate_flops_torchinfo(model, input_size=(1, 3, 224, 224)):
    """Calculate FLOPs using torchinfo"""
    try:
        model.eval()
        stats = summary(model, input_size=input_size, verbose=0)
        return {
            'flops_gmacs': getattr(stats, 'total_mult_adds', 0) / 1e9,
            'params_millions': getattr(stats, 'total_params', 0) / 1e6,
            'method': 'torchinfo'
        }
    except Exception as e:
        return {'error': f"torchinfo failed: {str(e)[:100]}"}

def calculate_flops_ptflops(model, input_size=(3, 224, 224)):
    """Calculate FLOPs using ptflops"""
    try:
        model.eval()
        macs, params = get_model_complexity_info(
            model, input_size, 
            as_strings=False, 
            print_per_layer_stat=False, 
            verbose=False
        )
        return {
            'flops_gmacs': macs / 1e9,
            'params_millions': params / 1e6,
            'method': 'ptflops'
        }
    except Exception as e:
        return {'error': f"ptflops failed: {str(e)[:100]}"}

def analyze_model_comprehensive(model_path, experiment_info):
    """Comprehensive model analysis"""
    results = {
        'experiment_id': experiment_info['id'],
        'model_path': str(model_path),
        'architecture': experiment_info['arch'],
        'experiment_type': experiment_info['type'],
        'load_status': 'Failed',
        'flops_gmacs': 'N/A',
        'params_millions': 'N/A',
        'model_size_mb': 'N/A',
        'analysis_method': 'None',
        'notes': ''
    }
    
    try:
        # Get file size
        results['model_size_mb'] = model_path.stat().st_size / (1024 * 1024)
        
        # Try to load the model
        model = None
        load_method = None
        
        # Method 1: Try JIT load first
        try:
            model = torch.jit.load(model_path, map_location=DEVICE)
            load_method = 'jit'
            results['load_status'] = 'Success (JIT)'
        except:
            pass
        
        # Method 2: Try loading as state dict
        if model is None:
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
                
                # Handle different checkpoint formats
                state_dict = None
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                elif isinstance(checkpoint, nn.Module):
                    model = checkpoint
                    load_method = 'direct_model'
                
                if state_dict is not None:
                    # Remove 'module.' prefix if present (DataParallel)
                    if any(key.startswith('module.') for key in state_dict.keys()):
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    
                    # Determine number of classes from the final layer
                    fc_key = None
                    for key in state_dict.keys():
                        if 'fc.' in key or 'classifier.' in key:
                            if 'weight' in key:
                                fc_key = key
                                break
                    
                    num_classes = 1000  # default
                    if fc_key and fc_key in state_dict:
                        num_classes = state_dict[fc_key].shape[0]
                    
                    # Create model and load weights
                    model = get_base_model(experiment_info['arch'], num_classes)
                    model.load_state_dict(state_dict, strict=False)
                    load_method = 'state_dict'
                
                if model is not None:
                    results['load_status'] = f'Success ({load_method})'
                    
            except Exception as e:
                results['notes'] = f"Load error: {str(e)[:100]}"
        
        # If model loaded successfully, analyze it
        if model is not None:
            model.to(DEVICE).eval()
            
            # Get basic model info
            model_info = extract_model_info(model)
            results['params_millions'] = model_info['total_params'] / 1e6
            
            # Try different FLOP calculation methods
            flop_results = None
            
            # For JIT models, we can't easily calculate FLOPs, so we'll note this
            if load_method == 'jit':
                results['flops_gmacs'] = 'N/A (JIT)'
                results['analysis_method'] = 'JIT (FLOPs unavailable)'
                results['notes'] = 'JIT model - FLOPs calculation not supported'
            else:
                # Try thop first
                if profiling_tools.get('thop', False):
                    flop_results = calculate_flops_thop(model)
                    if 'error' not in flop_results:
                        results.update(flop_results)
                        results['analysis_method'] = flop_results['method']
                
                # If thop failed, try torchinfo
                if (flop_results is None or 'error' in flop_results) and profiling_tools.get('torchinfo', False):
                    flop_results = calculate_flops_torchinfo(model)
                    if 'error' not in flop_results:
                        results.update(flop_results)
                        results['analysis_method'] = flop_results['method']
                
                # If both failed, try ptflops
                if (flop_results is None or 'error' in flop_results) and profiling_tools.get('ptflops', False):
                    flop_results = calculate_flops_ptflops(model)
                    if 'error' not in flop_results:
                        results.update(flop_results)
                        results['analysis_method'] = flop_results['method']
                
                # If all methods failed
                if flop_results is None or 'error' in flop_results:
                    results['analysis_method'] = 'All FLOP methods failed'
                    if flop_results and 'error' in flop_results:
                        results['notes'] = flop_results['error']
            
        # Clean up
        if model is not None:
            del model
        gc.collect()
        
    except Exception as e:
        results['notes'] = f"Analysis error: {str(e)[:100]}"
    
    return results

def parse_experiment_info(exp_path):
    """Extract experiment information from path and logs"""
    exp_name = exp_path.name
    parent_name = exp_path.parent.name
    
    # Determine architecture
    arch = 'resnet50'  # default
    if 'resnet18' in exp_name.lower():
        arch = 'resnet18'
    
    # Determine experiment type
    exp_type = 'unknown'
    if 'baseline' in exp_name.lower():
        exp_type = 'baseline'
    elif 'prune' in exp_name.lower():
        if 'struct' in exp_name.lower():
            exp_type = 'structured_pruning'
        elif 'unstruct' in exp_name.lower():
            exp_type = 'unstructured_pruning'
        elif 'nm' in exp_name.lower():
            exp_type = 'nm_sparsity'
        else:
            exp_type = 'pruning'
    elif 'quant' in exp_name.lower():
        exp_type = 'quantization'
    elif 'distill' in exp_name.lower():
        exp_type = 'knowledge_distillation'
    elif 'combined' in parent_name.lower():
        exp_type = 'combined_optimization'
    
    return {
        'id': f"{parent_name}/{exp_name}",
        'arch': arch,
        'type': exp_type,
        'name': exp_name,
        'parent': parent_name
    }

def main():
    print(f"--- Comprehensive Model Analysis Script ---")
    print(f"Models directory: {Path(MODELS_ROOT_DIR).resolve()}")
    print(f"Output CSV: {Path(OUTPUT_CSV).resolve()}")
    
    # Find all experiment directories with .pth files
    experiment_dirs = []
    for root, dirs, files in os.walk(MODELS_ROOT_DIR):
        if any(f.endswith('.pth') for f in files):
            experiment_dirs.append(Path(root))
    
    experiment_dirs = sorted(set(experiment_dirs))
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    all_results = []
    
    for exp_path in experiment_dirs:
        print(f"\nProcessing: {exp_path}")
        
        # Find model file
        model_files = list(exp_path.glob("*.pth"))
        if not model_files:
            continue
        
        # Prefer model_final.pth, otherwise take the first .pth file
        model_file = None
        for f in model_files:
            if 'final' in f.name:
                model_file = f
                break
        if model_file is None:
            model_file = model_files[0]
        
        print(f"  Model file: {model_file.name}")
        
        # Parse experiment info
        exp_info = parse_experiment_info(exp_path)
        
        # Analyze the model
        results = analyze_model_comprehensive(model_file, exp_info)
        all_results.append(results)
        
        print(f"  Status: {results['load_status']}")
        print(f"  FLOPs: {results['flops_gmacs']} GMACs")
        print(f"  Params: {results['params_millions']} M")
        print(f"  Size: {results['model_size_mb']:.1f} MB")
    
    # Create and save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Sort by experiment type and name
        df = df.sort_values(['experiment_type', 'experiment_id'])
        
        # Reorder columns for better readability
        column_order = [
            'experiment_id', 'experiment_type', 'architecture', 
            'flops_gmacs', 'params_millions', 'model_size_mb',
            'load_status', 'analysis_method', 'model_path', 'notes'
        ]
        df = df.reindex(columns=column_order)
        
        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False, float_format='%.5f')
        
        print(f"\n--- Results saved to {OUTPUT_CSV} ---")
        print(f"\nSummary:")
        print(f"Total experiments: {len(df)}")
        print(f"Successfully loaded: {len(df[df['load_status'].str.contains('Success', na=False)])}")
        print(f"FLOP calculations: {len(df[df['flops_gmacs'] != 'N/A'])}")
        
        # Show summary by experiment type
        print(f"\nBy experiment type:")
        type_summary = df.groupby('experiment_type').agg({
            'flops_gmacs': lambda x: f"{len(x[x != 'N/A'])}/{len(x)}",
            'params_millions': lambda x: f"{len(x[x != 'N/A'])}/{len(x)}"
        })
        print(type_summary)
        
        # Show first few rows
        print(f"\nFirst 5 results:")
        display_cols = ['experiment_id', 'experiment_type', 'flops_gmacs', 'params_millions', 'load_status']
        print(df[display_cols].head().to_string())
        
    else:
        print("No models found to analyze!")

if __name__ == "__main__":
    main()