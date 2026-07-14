"""Shared utilities for benchmarks and tests."""

import os
import json
import subprocess
from typing import Any, Dict, List, Union, Optional
from datetime import datetime

import psutil
import scipy.io.wavfile as wavfile

# Check for torch availability once at module level
TORCH_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass


def check_audio_file_is_silent(audio_path: str, threshold: float = 0.01) -> bool:
    """Check if an audio file is silent by comparing peak amplitude to a threshold.

    Args:
        audio_path: Path to the audio file
        threshold: Peak amplitude threshold for silence

    Returns:
        bool: True if audio is silent, False otherwise
    """
    rate, data = wavfile.read(audio_path)
    peak_amplitude = max(abs(data.min()), abs(data.max())) / 32768.0  # 16-bit audio

    return peak_amplitude < threshold


def get_audio_length(audio_data: bytes, temp_dir: str = None) -> float:
    """Get audio length in seconds from bytes data.

    Args:
        audio_data: Raw audio bytes
        temp_dir: Directory for temporary file. If None, uses system temp directory.

    Returns:
        float: Audio length in seconds
    """
    if temp_dir is None:
        import tempfile

        temp_dir = tempfile.gettempdir()

    temp_path = os.path.join(temp_dir, "temp.wav")
    os.makedirs(temp_dir, exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(audio_data)

    try:
        rate, data = wavfile.read(temp_path)
        return len(data) / rate
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_gpu_memory(average: bool = True) -> Optional[Union[float, List[float]]]:
    """Get GPU memory usage using PyTorch if available, falling back to nvidia-smi.

    Args:
        average: If True and multiple GPUs present, returns average memory usage.
                If False, returns list of memory usage per GPU.

    Returns:
        float or List[float] or None: GPU memory usage in MB. Returns None if no GPU available.
        If average=False and multiple GPUs present, returns list of values.
    """
    if TORCH_AVAILABLE:
        n_gpus = torch.cuda.device_count()
        memory_used = []
        for i in range(n_gpus):
            memory_used.append(
                torch.cuda.memory_allocated(i) / 1024**2
            )  # Convert to MB

        if average and len(memory_used) > 0:
            return sum(memory_used) / len(memory_used)
        return memory_used if len(memory_used) > 1 else memory_used[0]

    # Fall back to nvidia-smi
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        memory_values = [
            float(x.strip()) for x in result.decode("utf-8").split("\n") if x.strip()
        ]

        if average and len(memory_values) > 0:
            return sum(memory_values) / len(memory_values)
        return memory_values if len(memory_values) > 1 else memory_values[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_system_metrics() -> Dict[str, Union[str, float]]:
    """Get current system metrics including CPU, RAM, and GPU if available.

    Returns:
        dict: System metrics including timestamp, CPU%, RAM%, RAM GB, and GPU MB if available
    """
    # Get per-CPU percentages and calculate average
    cpu_percentages = psutil.cpu_percent(percpu=True)
    avg_cpu = sum(cpu_percentages) / len(cpu_percentages)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": round(avg_cpu, 2),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": psutil.virtual_memory().used / (1024**3),
    }

    gpu_mem = get_gpu_memory(average=True)  # Use average for system metrics
    if gpu_mem is not None:
        metrics["gpu_memory_used"] = round(gpu_mem, 2)

    return metrics


def save_audio_file(audio_data: bytes, identifier: str, output_dir: str) -> str:
    """Save audio data to a file with proper naming and directory creation.

    Args:
        audio_data: Raw audio bytes
        identifier: String to identify this audio file (e.g. token count, test name)
        output_dir: Directory to save the file

    Returns:
        str: Path to the saved audio file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{identifier}.wav")

    with open(output_file, "wb") as f:
        f.write(audio_data)

    return output_file


def write_benchmark_stats(stats: List[Dict[str, Any]], output_file: str) -> None:
    """Write benchmark statistics to a file in a clean, organized format.

    Args:
        stats: List of dictionaries containing stat name/value pairs
        output_file: Path to output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for section in stats:
            # Write section header
            f.write(f"=== {section['title']} ===\n\n")

            # Write stats
            for label, value in section["stats"].items():
                if isinstance(value, float):
                    f.write(f"{label}: {value:.2f}\n")
                else:
                    f.write(f"{label}: {value}\n")
            f.write("\n")


def save_json_results(results: Dict[str, Any], output_file: str) -> None:
    """Save benchmark results to a JSON file with proper formatting.

    Args:
        results: Dictionary of results to save
        output_file: Path to output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def real_time_factor(
    processing_time: float, audio_length: float, decimals: int = 2
) -> float:
    """Calculate Real-Time Factor (RTF) as processing-time / length-of-audio.

    Args:
        processing_time: Time taken to process/generate audio
        audio_length: Length of the generated audio
        decimals: Number of decimal places to round to

    Returns:
        float: RTF value
    """
    rtf = processing_time / audio_length
    return round(rtf, decimals)
