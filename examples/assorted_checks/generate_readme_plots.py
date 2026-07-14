#!/usr/bin/env python3
"""Script to generate all plots needed for the README."""

import os
import sys
import shutil
from pathlib import Path

from validate_wav import validate_tts

# Get absolute paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent

# Add directories to Python path for imports
sys.path.append(str(script_dir))
sys.path.append(str(script_dir / "benchmarks"))

# Import test scripts
from benchmark_tts_rtf import main as benchmark_rtf
from test_formats.test_audio_formats import main as test_formats
from benchmark_first_token_stream_unified import main as benchmark_stream
from test_combinations.test_analyze_combined_voices import main as test_voice_analysis

# Remove directories from path after imports
sys.path.remove(str(script_dir))
sys.path.remove(str(script_dir / "benchmarks"))


def ensure_assets_dir():
    """Create assets directory if it doesn't exist."""
    assets_dir = project_root / "assets"
    assets_dir.mkdir(exist_ok=True)
    return assets_dir


def copy_plot(src_path: str, dest_name: str, assets_dir: Path):
    """Copy a plot to the assets directory with a new name."""
    if os.path.exists(src_path):
        shutil.copy2(src_path, assets_dir / dest_name)
        print(f"Copied {src_path} to {assets_dir / dest_name}")
    else:
        print(f"Warning: Source plot not found at {src_path}")


def validate_and_print(wav_path: str, category: str):
    """Validate a WAV file and print results."""
    if not os.path.exists(wav_path):
        print(f"Warning: WAV file not found at {wav_path}")
        return

    print(f"\n=== Validating {category} Audio ===")
    result = validate_tts(wav_path)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Duration: {result['duration']}")
        print(f"Sample Rate: {result['sample_rate']} Hz")
        print(f"Peak Amplitude: {result['peak_amplitude']}")
        print(f"RMS Level: {result['rms_level']}")

        if result["issues"]:
            print("\nIssues Found:")
            for issue in result["issues"]:
                print(f"- {issue}")
        else:
            print("\nNo issues found")


def main():
    """Generate all plots needed for the README."""
    # Ensure assets directory exists
    prefix = "gpu"
    assets_dir = ensure_assets_dir()

    print("\n=== Generating Format Comparison Plot ===")
    test_formats()
    copy_plot(
        str(script_dir / "test_formats/output/test_formats/format_comparison.png"),
        "format_comparison.png",
        assets_dir,
    )
    # Validate WAV output from format test
    validate_and_print(
        str(script_dir / "test_formats/output/test_formats/speech.wav"),
        "Format Test WAV",
    )

    print("\n=== Generating Voice Analysis Plot ===")
    test_voice_analysis()
    copy_plot(
        str(script_dir / "test_combinations/output/analysis_comparison.png"),
        "voice_analysis.png",
        assets_dir,
    )
    # Validate combined voice output
    validate_and_print(
        str(
            script_dir
            / "test_combinations/output/analysis_combined_af_bella_af_nicole.wav"
        ),
        "Combined Voice",
    )

    print("\n=== Generating Performance Benchmark Plots ===")
    benchmark_rtf()
    copy_plot(
        str(script_dir / f"benchmarks/output_plots/{prefix}_processing_time_rtf.png"),
        f"{prefix}_processing_time.png",
        assets_dir,
    )
    copy_plot(
        str(script_dir / f"benchmarks/output_plots/{prefix}_realtime_factor_rtf.png"),
        f"{prefix}_realtime_factor.png",
        assets_dir,
    )
    # Validate RTF benchmark output (~500 tokens)
    validate_and_print(
        str(script_dir / "benchmarks/output_audio/chunk_450_tokens.wav"),
        "RTF Benchmark",
    )

    print("\n=== Generating Streaming Benchmark Plots ===")
    benchmark_stream()

    # Copy direct streaming plots
    copy_plot(
        str(script_dir / "benchmarks/output_plots/first_token_latency_stream.png"),
        f"{prefix}_first_token_latency_direct.png",
        assets_dir,
    )
    copy_plot(
        str(script_dir / "benchmarks/output_plots/first_token_timeline_stream.png"),
        f"{prefix}_first_token_timeline_direct.png",
        assets_dir,
    )
    copy_plot(
        str(script_dir / "benchmarks/output_plots/total_time_latency_stream.png"),
        f"{prefix}_total_time_latency_direct.png",
        assets_dir,
    )

    # Copy OpenAI streaming plots
    copy_plot(
        str(
            script_dir / "benchmarks/output_plots/first_token_latency_stream_openai.png"
        ),
        f"{prefix}_first_token_latency_openai.png",
        assets_dir,
    )
    copy_plot(
        str(
            script_dir
            / "benchmarks/output_plots/first_token_timeline_stream_openai.png"
        ),
        f"{prefix}_first_token_timeline_openai.png",
        assets_dir,
    )
    copy_plot(
        str(
            script_dir / "benchmarks/output_plots/total_time_latency_stream_openai.png"
        ),
        f"{prefix}_total_time_latency_openai.png",
        assets_dir,
    )

    # Wait a moment for files to be generated
    import time

    time.sleep(2)

    # Validate streaming outputs (~500 tokens)
    validate_and_print(
        str(
            script_dir
            / "benchmarks/output_audio_stream/benchmark_tokens500_run1_stream.wav"
        ),
        "Direct Streaming",
    )
    validate_and_print(
        str(
            script_dir
            / "benchmarks/output_audio_stream_openai/benchmark_tokens500_run1_stream_openai.wav"
        ),
        "OpenAI Streaming",
    )

    validate_and_print(
        str(script_dir / "test_formats/output/test_formats/test_audio.wav"),
        "Format Test WAV",
    )

    print("\nAll plots have been generated and copied to the assets directory")


if __name__ == "__main__":
    main()
