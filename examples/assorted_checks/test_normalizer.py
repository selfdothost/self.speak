import re
import time
import random
import string
from typing import List, Tuple


def create_test_cases() -> List[str]:
    """Create a variety of test cases with different characteristics"""

    # Helper to create random text with specific patterns
    def random_text(length: int) -> str:
        return "".join(
            random.choice(string.ascii_letters + string.digits + " .,!?")
            for _ in range(length)
        )

    test_cases = []

    # Base test cases that hit specific patterns
    base_cases = [
        "Dr. Smith and Mr. Jones discussed the $1,234.56 million investment.",
        "Yeah, they met at 10:30 and reviewed A.B.C. documentation with Mrs. Brown etc.",
        'The temperature was 72.5 degrees (quite normal) for "this time" of year.',
        "X's and Y's properties cost £50 million in the 1990s",
        "こんにちは。今日は！",
    ]

    # Add base cases
    test_cases.extend(base_cases)

    # Add variations with random content
    for length in [100, 1000, 10000]:
        # Create 3 variations of each length
        for _ in range(3):
            text = random_text(length)
            # Insert some patterns we're looking for
            text = text.replace(text[10:20], "Dr. Smith")
            text = text.replace(text[30:40], "$1,234.56")
            text = text.replace(text[50:60], "A.B.C. xyz")
            test_cases.append(text)

    return test_cases


class TextNormalizerInline:
    """Text normalizer using inline patterns"""

    def normalize(self, text: str) -> str:
        # Replace quotes and brackets
        text = text.replace(chr(8216), "'").replace(chr(8217), "'")
        text = text.replace("«", chr(8220)).replace("»", chr(8221))
        text = text.replace(chr(8220), '"').replace(chr(8221), '"')
        text = text.replace("(", "«").replace(")", "»")

        # Handle CJK punctuation
        for a, b in zip("、。！，：；？", ",.!,:;?"):
            text = text.replace(a, b + " ")

        text = re.sub(r"[^\S \n]", " ", text)
        text = re.sub(r"  +", " ", text)
        text = re.sub(r"(?<=\n) +(?=\n)", "", text)
        text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
        text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
        text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
        text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
        text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
        text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
        text = re.sub(
            r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)",
            split_num,
            text,
        )
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        text = re.sub(
            r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
            handle_money,
            text,
        )
        text = re.sub(r"\d*\.\d+", handle_decimal, text)
        text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
        text = re.sub(r"(?<=\d)S", " S", text)
        text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
        text = re.sub(r"(?<=X')S\b", "s", text)
        text = re.sub(
            r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text
        )
        text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)

        return text.strip()


class TextNormalizerCompiled:
    """Text normalizer using all compiled patterns"""

    def __init__(self):
        self.patterns = {
            "whitespace": re.compile(r"[^\S \n]"),
            "multi_space": re.compile(r"  +"),
            "newline_space": re.compile(r"(?<=\n) +(?=\n)"),
            "doctor": re.compile(r"\bD[Rr]\.(?= [A-Z])"),
            "mister": re.compile(r"\b(?:Mr\.|MR\.(?= [A-Z]))"),
            "miss": re.compile(r"\b(?:Ms\.|MS\.(?= [A-Z]))"),
            "mrs": re.compile(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))"),
            "etc": re.compile(r"\betc\.(?! [A-Z])"),
            "yeah": re.compile(r"(?i)\b(y)eah?\b"),
            "numbers": re.compile(
                r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)"
            ),
            "comma_in_number": re.compile(r"(?<=\d),(?=\d)"),
            "money": re.compile(
                r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b"
            ),
            "decimal": re.compile(r"\d*\.\d+"),
            "range": re.compile(r"(?<=\d)-(?=\d)"),
            "s_after_number": re.compile(r"(?<=\d)S"),
            "possessive_s": re.compile(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b"),
            "x_possessive": re.compile(r"(?<=X')S\b"),
            "initials": re.compile(r"(?:[A-Za-z]\.){2,} [a-z]"),
            "single_initial": re.compile(r"(?i)(?<=[A-Z])\.(?=[A-Z])"),
        }

    def normalize(self, text: str) -> str:
        # Replace quotes and brackets
        text = text.replace(chr(8216), "'").replace(chr(8217), "'")
        text = text.replace("«", chr(8220)).replace("»", chr(8221))
        text = text.replace(chr(8220), '"').replace(chr(8221), '"')
        text = text.replace("(", "«").replace(")", "»")

        # Handle CJK punctuation
        for a, b in zip("、。！，：；？", ",.!,:;?"):
            text = text.replace(a, b + " ")

        # Use compiled patterns
        text = self.patterns["whitespace"].sub(" ", text)
        text = self.patterns["multi_space"].sub(" ", text)
        text = self.patterns["newline_space"].sub("", text)
        text = self.patterns["doctor"].sub("Doctor", text)
        text = self.patterns["mister"].sub("Mister", text)
        text = self.patterns["miss"].sub("Miss", text)
        text = self.patterns["mrs"].sub("Mrs", text)
        text = self.patterns["etc"].sub("etc", text)
        text = self.patterns["yeah"].sub(r"\1e'a", text)
        text = self.patterns["numbers"].sub(split_num, text)
        text = self.patterns["comma_in_number"].sub("", text)
        text = self.patterns["money"].sub(handle_money, text)
        text = self.patterns["decimal"].sub(handle_decimal, text)
        text = self.patterns["range"].sub(" to ", text)
        text = self.patterns["s_after_number"].sub(" S", text)
        text = self.patterns["possessive_s"].sub("'S", text)
        text = self.patterns["x_possessive"].sub("s", text)
        text = self.patterns["initials"].sub(
            lambda m: m.group().replace(".", "-"), text
        )
        text = self.patterns["single_initial"].sub("-", text)

        return text.strip()


class TextNormalizerHybrid:
    """Text normalizer using hybrid approach - compile only complex/frequent patterns"""

    def __init__(self):
        # Only compile patterns that are complex or frequently used
        self.patterns = {
            "whitespace": re.compile(r"[^\S \n]"),
            "numbers": re.compile(
                r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)"
            ),
            "money": re.compile(
                r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b"
            ),
            "initials": re.compile(r"(?:[A-Za-z]\.){2,} [a-z]"),
        }

    def normalize(self, text: str) -> str:
        # Replace quotes and brackets
        text = text.replace(chr(8216), "'").replace(chr(8217), "'")
        text = text.replace("«", chr(8220)).replace("»", chr(8221))
        text = text.replace(chr(8220), '"').replace(chr(8221), '"')
        text = text.replace("(", "«").replace(")", "»")

        # Handle CJK punctuation
        for a, b in zip("、。！，：；？", ",.!,:;?"):
            text = text.replace(a, b + " ")

        # Use compiled patterns for complex operations
        text = self.patterns["whitespace"].sub(" ", text)
        text = self.patterns["numbers"].sub(split_num, text)
        text = self.patterns["money"].sub(handle_money, text)
        text = self.patterns["initials"].sub(
            lambda m: m.group().replace(".", "-"), text
        )

        # Use inline patterns for simpler operations
        text = re.sub(r"  +", " ", text)
        text = re.sub(r"(?<=\n) +(?=\n)", "", text)
        text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
        text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
        text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
        text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
        text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
        text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        text = re.sub(r"\d*\.\d+", handle_decimal, text)
        text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
        text = re.sub(r"(?<=\d)S", " S", text)
        text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
        text = re.sub(r"(?<=X')S\b", "s", text)
        text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)

        return text.strip()


def split_num(match: re.Match) -> str:
    """Split numbers for TTS processing"""
    num = match.group(0)
    if ":" in num:
        h, m = num.split(":")
        return f"{h} {m}"
    if num.endswith("s"):
        return f"{num[:-1]} s"
    return num


def handle_money(match: re.Match) -> str:
    """Format money strings for TTS"""
    text = match.group(0)
    return text.replace("$", " dollars ").replace("£", " pounds ")


def handle_decimal(match: re.Match) -> str:
    """Format decimal numbers for TTS"""
    num = match.group(0)
    return num.replace(".", " point ")


def benchmark_normalizers(
    test_cases: List[str], iterations: int = 100
) -> Tuple[float, float, float]:
    """Benchmark all three implementations"""

    normalizers = {
        "inline": TextNormalizerInline(),
        "compiled": TextNormalizerCompiled(),
        "hybrid": TextNormalizerHybrid(),
    }

    results = {}

    # Test each normalizer
    for name, normalizer in normalizers.items():
        start = time.perf_counter()

        # Run normalizations
        for _ in range(iterations):
            for test in test_cases:
                normalizer.normalize(test)

        results[name] = time.perf_counter() - start

    return results


def verify_outputs(test_cases: List[str]) -> bool:
    """Verify that all implementations produce identical output"""
    normalizers = {
        "inline": TextNormalizerInline(),
        "compiled": TextNormalizerCompiled(),
        "hybrid": TextNormalizerHybrid(),
    }

    for test in test_cases:
        results = [norm.normalize(test) for norm in normalizers.values()]
        if not all(r == results[0] for r in results):
            return False
    return True


def main():
    # Create test cases
    print("Generating test cases...")
    test_cases = create_test_cases()
    total_chars = sum(len(t) for t in test_cases)
    print(
        f"Created {len(test_cases)} test cases, total size: {total_chars:,} characters"
    )

    # Verify output consistency
    print("\nVerifying output consistency...")
    if verify_outputs(test_cases):
        print("✓ All implementations produce identical output")
    else:
        print("✗ Warning: Implementations produce different outputs!")
        return

    # Run benchmarks
    print("\nRunning benchmarks...")
    iterations = 100
    results = benchmark_normalizers(test_cases, iterations)

    # Print results
    print(f"\nResults for {iterations} iterations: ")
    for name, time_taken in results.items():
        print(f"{name.capitalize()}: {time_taken:.3f}s")


main()
