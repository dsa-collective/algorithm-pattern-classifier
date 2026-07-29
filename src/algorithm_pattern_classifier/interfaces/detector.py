import ast
from abc import ABC, abstractmethod

from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class BaseDetector(ABC):
    """Abstract base class for all algorithmic pattern detectors."""

    @property
    @abstractmethod
    def pattern(self) -> AlgorithmPattern:
        """The AlgorithmPattern that this detector is designed to detect."""
        pass

    @abstractmethod
    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Detect evidence of the specific pattern in the AST.

        Args:
            code_ast: The parsed Abstract Syntax Tree (AST) of the source code.

        Returns:
            A PatternMatch object if the pattern is detected, or None otherwise.
        """
        pass
