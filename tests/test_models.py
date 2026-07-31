import ast
from dataclasses import FrozenInstanceError

import pytest

from algorithm_pattern_classifier.interfaces.classifier import BaseClassifier
from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


def test_algorithm_pattern_enum() -> None:
    """Test AlgorithmPattern enum members and values."""
    assert AlgorithmPattern.TWO_POINTERS.value == "two-pointers"
    assert AlgorithmPattern.SLIDING_WINDOW.value == "sliding-window"
    assert AlgorithmPattern.DYNAMIC_PROGRAMMING.value == "dynamic-programming"
    assert AlgorithmPattern.BACKTRACKING.value == "backtracking"
    assert AlgorithmPattern.FAST_SLOW_POINTERS.value == "fast-slow-pointers"
    assert AlgorithmPattern.DIVIDE_AND_CONQUER.value == "divide-and-conquer"
    assert AlgorithmPattern.BFS.value == "bfs"
    assert AlgorithmPattern.DFS.value == "dfs"
    assert AlgorithmPattern.GREEDY.value == "greedy"


def test_pattern_match_construction_and_equality() -> None:
    """Test PatternMatch construction and value equality."""
    result1 = PatternMatch(
        pattern=AlgorithmPattern.TWO_POINTERS,
        confidence=0.85,
        evidence=["line 10: left, right pointers initialized"],
    )
    result2 = PatternMatch(
        pattern=AlgorithmPattern.TWO_POINTERS,
        confidence=0.85,
        evidence=["line 10: left, right pointers initialized"],
    )

    assert result1 == result2
    assert result1.pattern == AlgorithmPattern.TWO_POINTERS
    assert result1.confidence == 0.85
    assert len(result1.evidence) == 1


def test_pattern_match_defaults() -> None:
    """Test that PatternMatch defaults evidence to an empty list."""
    result = PatternMatch(
        pattern=AlgorithmPattern.TWO_POINTERS,
        confidence=0.85,
    )
    assert result.evidence == []


def test_pattern_match_immutability() -> None:
    """Test that PatternMatch is frozen and cannot be mutated."""
    result = PatternMatch(
        pattern=AlgorithmPattern.TWO_POINTERS,
        confidence=0.85,
    )
    with pytest.raises(FrozenInstanceError):
        result.confidence = 0.95  # type: ignore[misc]


def test_base_detector_cannot_be_instantiated() -> None:
    """Verify that abstract class BaseDetector cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseDetector()  # type: ignore[abstract]


def test_base_classifier_cannot_be_instantiated() -> None:
    """Verify that abstract class BaseClassifier cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseClassifier()  # type: ignore[abstract]


def test_interfaces_implementability() -> None:
    """Verify that BaseDetector and BaseClassifier can be concretely subclassed."""

    class MockDetector(BaseDetector):
        @property
        def pattern(self) -> AlgorithmPattern:
            return AlgorithmPattern.TWO_POINTERS

        def detect(self, _code_ast: ast.AST) -> PatternMatch | None:
            return PatternMatch(
                pattern=AlgorithmPattern.TWO_POINTERS,
                confidence=1.0,
                evidence=["found while loop with left < right"],
            )

    class MockClassifier(BaseClassifier):
        def classify(self, source_code: str) -> list[PatternMatch]:
            _ = source_code
            return [
                PatternMatch(
                    pattern=AlgorithmPattern.TWO_POINTERS,
                    confidence=1.0,
                    evidence=["mocked"],
                )
            ]

    detector = MockDetector()
    det_res = detector.detect(ast.parse("while left < right:\n    pass"))
    assert det_res is not None
    assert det_res.confidence == 1.0

    classifier = MockClassifier()
    cls_res = classifier.classify("code")
    assert len(cls_res) == 1
    assert cls_res[0].pattern == AlgorithmPattern.TWO_POINTERS
