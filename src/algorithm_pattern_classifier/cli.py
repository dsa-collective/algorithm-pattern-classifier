# ruff: noqa: T201
import argparse
import json
import sys
from pathlib import Path

from algorithm_pattern_classifier.classifiers.pattern_classifier import PatternClassifier


def main() -> None:
    """CLI entry point for classifying algorithm files into design patterns."""
    parser = argparse.ArgumentParser(
        description="Classify algorithm source code files into design patterns."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Classify subcommand
    classify_parser = subparsers.add_parser("classify", help="Classify a file.")
    classify_parser.add_argument("file", type=str, help="Path to the python file to analyze.")
    classify_parser.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON format."
    )

    args = parser.parse_args()

    if args.command == "classify":
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{args.file}' does not exist.", file=sys.stderr)
            sys.exit(1)

        try:
            source_code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)

        classifier = PatternClassifier()
        results = classifier.classify(source_code)

        if args.json:
            json_results = [
                {
                    "pattern": res.pattern.value,
                    "confidence_score": res.confidence,
                    "supporting_evidence": res.evidence,
                }
                for res in results
            ]
            print(json.dumps(json_results, indent=2))
        else:
            print("Algorithm Pattern Classification Report")
            print("=" * 40)
            print(f"File: {file_path}")
            print()
            if not results:
                print("No patterns detected.")
            else:
                print("Detected Patterns:")
                for res in results:
                    print(f"  - Pattern:    {res.pattern.value.upper()}")
                    print(f"    Confidence: {res.confidence * 100:.1f}%")
                    if res.evidence:
                        print("    Evidence:")
                        for evidence in res.evidence:
                            print(f"      * {evidence}")
                    print()


if __name__ == "__main__":
    main()
