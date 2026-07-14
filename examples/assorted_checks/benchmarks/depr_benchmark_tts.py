import os
import json
import time

import pandas as pd

from examples.assorted_checks.lib.shared_utils import (
    save_json_results,
    get_system_metrics,
    write_benchmark_stats,
)
from examples.assorted_checks.lib.shared_plotting import (
    plot_correlation,
    plot_system_metrics,
)
from examples.assorted_checks.lib.shared_benchmark_utils import (
    enc,
    make_tts_request,
    get_text_for_tokens,
    generate_token_sizes,
)


def main():
    # Get optional prefix from first command line argument
    import sys

    prefix = sys.argv[1] if len(sys.argv) > 1 else ""

    # Set up paths relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output_audio")
    output_data_dir = os.path.join(script_dir, "output_data")
    output_plots_dir = os.path.join(script_dir, "output_plots")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    # Function to prefix filenames
    def prefix_path(path: str, filename: str) -> str:
        if prefix:
            filename = f"{prefix}_{filename}"
        return os.path.join(path, filename)

    # Read input text
    with open(
        os.path.join(script_dir, "the_time_machine_hg_wells.txt"), "r", encoding="utf-8"
    ) as f:
        text = f.read()

    # Get total tokens in file
    total_tokens = len(enc.encode(text))
    print(f"Total tokens in file: {total_tokens}")

    token_sizes = generate_token_sizes(total_tokens)

    print(f"Testing sizes: {token_sizes}")

    # Process chunks
    results = []
    system_metrics = []
    test_start_time = time.time()

    for num_tokens in token_sizes:
        # Get text slice with exact token count
        chunk = get_text_for_tokens(text, num_tokens)
        actual_tokens = len(enc.encode(chunk))

        print(f"\nProcessing chunk with {actual_tokens} tokens:")
        print(f"Text preview: {chunk[:100]}...")

        # Collect system metrics before processing
        system_metrics.append(get_system_metrics())

        processing_time, audio_length = make_tts_request(chunk)
        if processing_time is None or audio_length is None:
            print("Breaking loop due to error")
            break

        # Collect system metrics after processing
        system_metrics.append(get_system_metrics())

        results.append(
            {
                "tokens": actual_tokens,
                "processing_time": processing_time,
                "output_length": audio_length,
                "realtime_factor": audio_length / processing_time,
                "elapsed_time": time.time() - test_start_time,
            }
        )

        # Save intermediate results
        save_json_results(
            {"results": results, "system_metrics": system_metrics},
            prefix_path(output_data_dir, "benchmark_results.json"),
        )

    # Create DataFrame and calculate stats
    df = pd.DataFrame(results)
    if df.empty:
        print("No data to plot")
        return

    # Calculate useful metrics
    df["tokens_per_second"] = df["tokens"] / df["processing_time"]

    # Write benchmark stats
    stats = [
        {
            "title": "Benchmark Statistics",
            "stats": {
                "Total tokens processed": df["tokens"].sum(),
                "Total audio generated (s)": df["output_length"].sum(),
                "Total test duration (s)": df["elapsed_time"].max(),
                "Average processing rate (tokens/s)": df["tokens_per_second"].mean(),
                "Average realtime factor": df["realtime_factor"].mean(),
            },
        },
        {
            "title": "Per-chunk Stats",
            "stats": {
                "Average chunk size (tokens)": df["tokens"].mean(),
                "Min chunk size (tokens)": df["tokens"].min(),
                "Max chunk size (tokens)": df["tokens"].max(),
                "Average processing time (s)": df["processing_time"].mean(),
                "Average output length (s)": df["output_length"].mean(),
            },
        },
        {
            "title": "Performance Ranges",
            "stats": {
                "Processing rate range (tokens/s)": f"{df['tokens_per_second'].min():.2f} - {df['tokens_per_second'].max():.2f}",
                "Realtime factor range": f"{df['realtime_factor'].min():.2f}x - {df['realtime_factor'].max():.2f}x",
            },
        },
    ]
    write_benchmark_stats(stats, prefix_path(output_data_dir, "benchmark_stats.txt"))

    # Plot Processing Time vs Token Count
    plot_correlation(
        df,
        "tokens",
        "processing_time",
        "Processing Time vs Input Size",
        "Number of Input Tokens",
        "Processing Time (seconds)",
        prefix_path(output_plots_dir, "processing_time.png"),
    )

    # Plot Realtime Factor vs Token Count
    plot_correlation(
        df,
        "tokens",
        "realtime_factor",
        "Realtime Factor vs Input Size",
        "Number of Input Tokens",
        "Realtime Factor (output length / processing time)",
        prefix_path(output_plots_dir, "realtime_factor.png"),
    )

    # Plot system metrics
    plot_system_metrics(
        system_metrics, prefix_path(output_plots_dir, "system_usage.png")
    )

    print("\nResults saved to:")
    print(f"- {prefix_path(output_data_dir, 'benchmark_results.json')}")
    print(f"- {prefix_path(output_data_dir, 'benchmark_stats.txt')}")
    print(f"- {prefix_path(output_plots_dir, 'processing_time.png')}")
    print(f"- {prefix_path(output_plots_dir, 'realtime_factor.png')}")
    print(f"- {prefix_path(output_plots_dir, 'system_usage.png')}")
    if any("gpu_memory_used" in m for m in system_metrics):
        print(f"- {prefix_path(output_plots_dir, 'gpu_usage.png')}")
    print(f"\nAudio files saved in {output_dir} with prefix: {prefix or '(none)'}")


if __name__ == "__main__":
    main()
