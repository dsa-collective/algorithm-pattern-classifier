import ast

from algorithm_pattern_classifier.classifiers.pattern_classifier import PatternClassifier
from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class MockDetector(BaseDetector):
    def __init__(self, pattern: AlgorithmPattern, confidence: float) -> None:
        self._pattern = pattern
        self.confidence_value = confidence

    @property
    def pattern(self) -> AlgorithmPattern:
        return self._pattern

    def detect(self, _code_ast: ast.AST) -> PatternMatch | None:
        if self.confidence_value <= 0.0:
            return None
        return PatternMatch(
            pattern=self._pattern,
            confidence=self.confidence_value,
            evidence=[f"Mock {self._pattern.value}"],
        )


def test_classifier_single_clear_match() -> None:
    """Test classifier returns a single match when only one detector fires."""
    detectors: list[BaseDetector] = [
        MockDetector(AlgorithmPattern.TWO_POINTERS, 0.95),
        MockDetector(AlgorithmPattern.SLIDING_WINDOW, 0.0),
    ]
    classifier = PatternClassifier(detectors=detectors)
    results = classifier.classify("pass")

    assert len(results) == 1
    assert results[0].pattern == AlgorithmPattern.TWO_POINTERS
    assert results[0].confidence == 0.95


def test_classifier_multiple_matches_ranked() -> None:
    """Test classifier returns multiple matches ranked by descending confidence."""
    detectors: list[BaseDetector] = [
        MockDetector(AlgorithmPattern.DYNAMIC_PROGRAMMING, 0.7),
        MockDetector(AlgorithmPattern.BFS, 0.9),
    ]
    classifier = PatternClassifier(detectors=detectors)
    results = classifier.classify("pass")

    assert len(results) == 2
    assert results[0].pattern == AlgorithmPattern.BFS
    assert results[0].confidence == 0.9
    assert results[1].pattern == AlgorithmPattern.DYNAMIC_PROGRAMMING
    assert results[1].confidence == 0.7


def test_classifier_no_match() -> None:
    """Test classifier returns empty list when no detectors fire."""
    detectors: list[BaseDetector] = [
        MockDetector(AlgorithmPattern.TWO_POINTERS, 0.0),
        MockDetector(AlgorithmPattern.SLIDING_WINDOW, 0.0),
    ]
    classifier = PatternClassifier(detectors=detectors)
    results = classifier.classify("pass")

    assert results == []


def test_classifier_overlap_handling() -> None:
    """Test classifier handles sliding-window and two-pointer overlap sensibly."""
    # Case 1: Equal confidence (sliding-window suppresses two-pointer)
    detectors1: list[BaseDetector] = [
        MockDetector(AlgorithmPattern.TWO_POINTERS, 0.8),
        MockDetector(AlgorithmPattern.SLIDING_WINDOW, 0.8),
    ]
    classifier1 = PatternClassifier(detectors=detectors1)
    results1 = classifier1.classify("pass")

    assert len(results1) == 1
    assert results1[0].pattern == AlgorithmPattern.SLIDING_WINDOW

    # Case 2: Two-pointer has higher confidence (no suppression)
    detectors2: list[BaseDetector] = [
        MockDetector(AlgorithmPattern.TWO_POINTERS, 0.9),
        MockDetector(AlgorithmPattern.SLIDING_WINDOW, 0.7),
    ]
    classifier2 = PatternClassifier(detectors=detectors2)
    results2 = classifier2.classify("pass")

    assert len(results2) == 2
    assert results2[0].pattern == AlgorithmPattern.TWO_POINTERS
    assert results2[1].pattern == AlgorithmPattern.SLIDING_WINDOW


def test_classifier_dp_suppresses_dfs() -> None:
    """Test that dynamic programming suppresses DFS when confidence is high enough."""
    detectors: list[BaseDetector] = [
        # Note: Dynamic programming is more specific than DFS. Under default rules,
        # a DP match with confidence >= 0.5 completely suppresses DFS,
        # meaning specificity deliberately outranks raw confidence.
        MockDetector(AlgorithmPattern.DYNAMIC_PROGRAMMING, 0.8),
        MockDetector(AlgorithmPattern.DFS, 0.9),
    ]
    classifier = PatternClassifier(detectors=detectors)
    results = classifier.classify("pass")

    # DFS should be suppressed since DP confidence >= 0.5
    assert len(results) == 1
    assert results[0].pattern == AlgorithmPattern.DYNAMIC_PROGRAMMING
