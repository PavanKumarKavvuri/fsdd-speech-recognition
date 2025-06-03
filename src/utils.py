import random
import numpy as np
import torch
import math
import os
from typing import List, Tuple, Any
import tempfile
from torch.nn import Module
from collections import defaultdict
import yaml
from torch.utils.benchmark import Timer

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across `random`, `numpy`, and `torch`.

    Args:
        seed (int): The seed value to use. Default is 42.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Random seed set to: {seed}")

def prepare_data_label_pairs(root_dir: str, calibration_samples_per_class: int = 10) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Scans a directory for .wav files and returns:
    1. A list of all (audio_path, label) pairs
    2. A filtered list containing up to N samples per digit class

    Assumes filenames are in the format: <digit>_<speaker>_<id>.wav

    Args:
        root_dir (str): Path to the directory containing audio files.
        samples_per_class (int): Number of samples to collect per digit class in the limited list.

    Returns:
        Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]: 
            - Full list of (file_path, label)
            - Limited list with N samples per class
    """
    data = []
    limited_data = []
    label_counts = defaultdict(int)

    for filename in sorted(os.listdir(root_dir)):
        if filename.endswith('.wav'):
            parts = filename.split('_')
            if len(parts) < 3:
                print(f"[WARNING] Skipping file with unexpected format: {filename}")
                continue
            try:
                label = int(parts[0])
                path = os.path.join(root_dir, filename)
                data.append((path, label))  # ✅ always populate the full list

                # Populate limited_data only up to N samples/class
                if 0 <= label <= 9 and label_counts[label] < calibration_samples_per_class:
                    limited_data.append((path, label))
                    label_counts[label] += 1

            except ValueError:
                print(f"[WARNING] Skipping file with invalid label: {filename}")

    return data, limited_data


def get_model_size_in_kb(model: Module) -> float:
    """
    Calculates the size of a PyTorch model in kilobytes.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.

    Returns:
        float: Model size in kilobytes (KB).
    """
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        model_size_kb = os.path.getsize(tmp.name) / 1024
    return model_size_kb

def get_model_params_size(model: Module):
    layer_params = defaultdict(int)

    for name, param in model.named_parameters():
        layer_params[name] += param.numel()

    total_params = sum(param.numel() for param in model.parameters())
    total_memory_bytes = total_params * 4 
    total_memory_kb = total_memory_bytes / 1024
    total_memory_mb = total_memory_kb / 1024

    # Optionally pretty print each layer
    print("\nLayer-wise parameter counts:")
    for layer, count in layer_params.items():
        print(f"{layer:30} -> {count:,} params")

    print(f"\n\n Total Parameters: {total_params:,}")
    print(f"Estimated Memory: {total_memory_kb:.2f} KB ({total_memory_mb:.2f} MB)")

    return dict(layer_params), total_memory_kb

def read_yaml_file(file_path: str) -> Any:
    """
    Reads a YAML file and returns its content as a Python dictionary or list.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Any: Parsed YAML content (usually dict or list).
    """
    try:
        with open(file_path, "r") as f:
            yaml_content = yaml.safe_load(f)
        return yaml_content
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except yaml.YAMLError as e:
        print(f"⚠️ YAML parsing error in {file_path}:\n{e}")
    except Exception as e:
        print(f"🚨 Unexpected error reading {file_path}:\n{e}")

def print_quantized_layer_analysis(model, model_name="Model"):
    print(f"\n {model_name} - Layer Analysis")
    print("Layer Name".ljust(40) + " | Num Parameters | Size (INT8) | Fits in 36KB?")
    print("-" * 75)

    total_params = 0
    total_kb = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Linear):
            weight = module.weight()
            if hasattr(weight, 'is_quantized') and weight.is_quantized:
                num_params = weight.numel()
                kb = (num_params * 1) / 1024  # int8 → 1 byte per param
                total_params += num_params
                total_kb += kb
                fits = kb <= 36.0
                print(f"{name.ljust(40)} | {str(num_params).rjust(13)} | {kb:.3f} KB     | {'✅' if fits else '❌'}")

    print("\n📦 Total Estimated Memory Usage")
    print(f"Total number of parameters:      {total_params}")
    print(f"Estimated total size (INT8):     {total_kb:.3f} KB")
    print(f"Memory per parameter (INT8):     1 byte")
    print(f"Meets 36KB per-layer limit?      {'✅ Yes' if total_kb <= 36.0 else '❌ No'}")


def print_float_model_analysis(model):
    print("Layer Name".ljust(25) + " | Num Parameters | Size (Memory)")
    print("-" * 70)

    total_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            num_params = param.numel()
            kb = (num_params * param.element_size()) / 1024
            total_params += num_params
            # fits = kb <= 36.0
            print(f"{name.ljust(25)} | {str(num_params).rjust(13)} | {kb:.3f} KB ")

    # Final summary
    total_kb = (total_params * 4) / 1024  # FP32 uses 4 bytes per param
    print("\n📊 Total Model Summary")
    print(f"Total number of parameters:      {total_params}")
    print(f"Estimated total size (FP32):    {total_kb:.2f} KB ({total_kb / 1024:.2f} MB)")
    print(f"Memory per parameter (FP32):    4 bytes")
    print(f"Meets 36KB per-layer limit?     {'✅ Yes' if total_kb <= 36.0 else '❌ No'}")


def count_quantized_weights(model):
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Linear):
            weight = module.weight()
            if weight.is_quantized:
                num_params = weight.numel()
                print(f"{name} → {num_params} params")
                total_params += num_params
    print(f"\n✅ Total quantized parameters: {total_params}")

def is_power_of_two(x, tolerance=1e-6):
    if x <= 0:
        return False
    log2_x = math.log2(x)
    return abs(log2_x - round(log2_x)) < tolerance

def verify_power_of_two_scales(model):
    print("Layer Name".ljust(35) + " | Scale Value     | Is Power of Two?")
    print("-" * 70)

    all_layers_are_power_of_two = True

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Linear):
            weight = module.weight()
            if hasattr(weight, 'is_quantized') and weight.is_quantized:
                qscheme = weight.qscheme()

                # Handle per-tensor symmetric or affine
                if qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                    scale = weight.q_scale()
                    pot = is_power_of_two(scale)
                    if not pot:
                        all_layers_are_power_of_two = False
                    print(f"{name.ljust(35)} | {scale:.8f}      | {'✅' if pot else '❌'}")

                # Handle per-channel scales
                elif qscheme == torch.per_channel_affine:
                    scales = weight.q_per_channel_scales()
                    all_pots = all(is_power_of_two(s.item()) for s in scales)
                    print(f"{name.ljust(35)} | {str(scales.tolist())[:40]}... | {'✅' if all_pots else '❌'}")
                    if not all_pots:
                        all_layers_are_power_of_two = False

                else:
                    print(f"{name.ljust(35)} | Unknown scheme: {qscheme}")

    print("\n Final Result")
    if all_layers_are_power_of_two:
        print("✅ All layers use power-of-two scale values.")
    else:
        print("❌ Some layers do NOT use power-of-two scale values.")

    return all_layers_are_power_of_two



def compute_inference_time(model: Module, test_loader: torch.utils.data.DataLoader) -> None:
    """
    Computes and prints the inference time for a given model on a test dataset.

    Args:
        model (torch.nn.Module): The model to benchmark.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        None
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Move model to CPU (required for quantized models, good practice for float too)
    model = model.to('cpu')

    # Get one sample from test_loader
    for input_batch, _ in test_loader:
        input_tensor = input_batch[0].unsqueeze(0).to('cpu')  # shape: (1, seq_len, feature_dim)
        break

    # Benchmark inference time
    timer = Timer(
        stmt="model(input_tensor)",
        globals={"model": model, "input_tensor": input_tensor},
        num_threads=torch.get_num_threads()
    )

    measurement = timer.blocked_autorange()
    print(f"Median inference time: {measurement.median * 1e3:.4f} ms")

