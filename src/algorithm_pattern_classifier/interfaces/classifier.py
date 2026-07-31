from abc import ABC, abstractmethod

from algorithm_pattern_classifier.models.patterns import PatternMatch


class BaseClassifier(ABC):
    """Abstract base class for algorithm pattern classifiers."""

    @abstractmethod
    def classify(self, source_code: str) -> list[PatternMatch]:
        """Classify and rank the algorithmic design patterns found in the source code.

        Args:
            source_code: The raw source code of the solution.

        Returns:
            A list of PatternMatch objects ranked by confidence.
        """
        pass
