from dataclasses import dataclass, field
from enum import StrEnum


class AlgorithmPattern(StrEnum):
    """Supported algorithmic design patterns for classification."""

    SLIDING_WINDOW = "sliding-window"
    TWO_POINTERS = "two-pointers"
    FAST_SLOW_POINTERS = "fast-slow-pointers"
    DIVIDE_AND_CONQUER = "divide-and-conquer"
    BACKTRACKING = "backtracking"
    BFS = "bfs"
    DFS = "dfs"
    DYNAMIC_PROGRAMMING = "dynamic-programming"
    GREEDY = "greedy"


@dataclass(frozen=True)
class PatternMatch:
    """Represents a detected design pattern with confidence and evidence."""

    pattern: AlgorithmPattern
    confidence: float
    evidence: list[str] = field(default_factory=list)
