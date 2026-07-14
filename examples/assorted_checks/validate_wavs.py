import argparse
from pathlib import Path

from validate_wav import validate_tts


def print_validation_result(result: dict, rel_path: Path):
    """Print full validation details for a single file."""
    print(f"\nValidating: {rel_path}")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Duration: {result['duration']}")
        print(f"Sample Rate: {result['sample_rate']} Hz")
        print(f"Peak Amplitude: {result['peak_amplitude']}")
        print(f"RMS Level: {result['rms_level']}")
        print(f"DC Offset: {result['dc_offset']}")

        if result["issues"]:
            print("\nIssues Found:")
            for issue in result["issues"]:
                print(f"- {issue}")
        else:
            print("\nNo issues found")


def validate_directory(directory: str):
    """Validate all wav files in a directory with detailed output and summary."""
    dir_path = Path(directory)

    # Find all wav files (including nested directories)
    wav_files = list(dir_path.rglob("*.wav"))
    wav_files.extend(dir_path.rglob("*.mp3"))  # Also check mp3s
    wav_files = sorted(wav_files)

    if not wav_files:
        print(f"No .wav or .mp3 files found in {directory}")
        return

    print(f"Found {len(wav_files)} files in {directory}")
    print("=" * 80)

    # Store results for summary
    results = []

    # Detailed validation output
    for wav_file in wav_files:
        result = validate_tts(str(wav_file))
        rel_path = wav_file.relative_to(dir_path)
        print_validation_result(result, rel_path)
        results.append((rel_path, result))
        print("=" * 80)

    # Summary with detailed issues
    print("\nSUMMARY:")
    for rel_path, result in results:
        if "error" in result:
            print(f"{rel_path}: ERROR - {result['error']}")
        elif result["issues"]:
            # Show first issue in summary, indicate if there are more
            issues = result["issues"]
            first_issue = issues[0].replace("WARNING: ", "")
            if len(issues) > 1:
                print(
                    f"{rel_path}: FAIL - {first_issue} (+{len(issues)-1} more issues)"
                )
            else:
                print(f"{rel_path}: FAIL - {first_issue}")
        else:
            print(f"{rel_path}: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch validate TTS wav files")
    parser.add_argument("directory", help="Directory containing wav files to validate")
    args = parser.parse_args()

    validate_directory(args.directory)
