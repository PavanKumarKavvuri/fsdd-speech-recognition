import random
import numpy as np
import torch
import os
from typing import List, Tuple, Any
import tempfile
from torch.nn import Module
from collections import defaultdict
import yaml

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


def prepare_data_label_pairs(root_dir: str) -> List[Tuple[str, int]]:
    """
    Scans a directory for .wav files and returns a list of (audio_path, label) tuples.

    Assumes filenames are in the format: <label>_<otherinfo>.wav

    Args:
        root_dir (str): Path to the directory containing audio files.

    Returns:
        List[Tuple[str, int]]: List of (file_path, label) pairs.
    """
    data = []
    for filename in os.listdir(root_dir):
        if filename.endswith('.wav'):
            try:
                label = int(filename.split('_')[0])  # Extract label before first underscore
                path = os.path.join(root_dir, filename)
                data.append((path, label))
            except ValueError:
                print(f"[WARNING] Skipping file with invalid label format: {filename}")
    return data


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

