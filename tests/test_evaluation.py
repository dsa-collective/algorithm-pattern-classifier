import sys
from pathlib import Path

from algorithm_pattern_classifier.classifiers.pattern_classifier import PatternClassifier
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern


def test_corpus_regression_evaluation() -> None:
    """Evaluate classifier against the labeled corpus and report precision/recall."""
    corpus_root = Path(__file__).parent / "corpus"
    assert corpus_root.exists(), "Corpus directory must exist"

    classifier = PatternClassifier()

    # Map directory names to expected pattern enums
    label_map = {
        "two_pointer": AlgorithmPattern.TWO_POINTERS,
        "sliding_window": AlgorithmPattern.SLIDING_WINDOW,
        "dynamic_programming": AlgorithmPattern.DYNAMIC_PROGRAMMING,
        "negatives": None,
    }

    total_runs = 0
    correct_runs = 0

    # For precision/recall tracking per pattern class
    # True Positives, False Positives, False Negatives
    tp: dict[AlgorithmPattern, int] = dict.fromkeys(AlgorithmPattern, 0)
    fp: dict[AlgorithmPattern, int] = dict.fromkeys(AlgorithmPattern, 0)
    fn: dict[AlgorithmPattern, int] = dict.fromkeys(AlgorithmPattern, 0)

    for label_dir, expected_pattern in label_map.items():
        dir_path = corpus_root / label_dir
        if not dir_path.exists():
            continue

        for file_path in dir_path.glob("*.py"):
            total_runs += 1
            source_code = file_path.read_text(encoding="utf-8")
            results = classifier.classify(source_code)

            # Get the top classification result if confidence is high enough
            top_result = None
            if results and results[0].confidence >= 0.5:
                top_result = results[0].pattern

            sys.stdout.write(
                f"File: {file_path.name} | Expected: {expected_pattern} | Got: {top_result}\n"
            )

            if top_result == expected_pattern:
                correct_runs += 1
                if expected_pattern is not None:
                    tp[expected_pattern] += 1
            else:
                # Mismatch / error
                if expected_pattern is not None:
                    # Expected a pattern but didn't get it (or got wrong one)
                    fn[expected_pattern] += 1
                if top_result is not None:
                    # Got a pattern when not expected (or wrong one)
                    fp[top_result] += 1

    # Print summary reports to stdout (will be visible on failure or with pytest -s)
    sys.stdout.write("\n" + "=" * 50 + "\n")
    sys.stdout.write("LABELED EVALUATION CORPUS REPORT\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.write(f"Total evaluated files: {total_runs}\n")
    sys.stdout.write(f"Overall Accuracy:      {correct_runs / total_runs * 100:.1f}%\n")
    sys.stdout.write("-" * 50 + "\n")

    # We must have 100% correct classifications on the reference corpus to prevent regressions
    accuracy = correct_runs / total_runs
    sys.stdout.write(f"Final accuracy: {accuracy * 100:.1f}%\n")

    # Assert accuracy to fail CI on regression
    err_msg = f"Regression detected! Labeled corpus accuracy is only {accuracy * 100:.1f}%"
    assert accuracy >= 1.0, err_msg
