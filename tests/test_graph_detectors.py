import ast

from algorithm_pattern_classifier.detectors.bfs import BFSDetector
from algorithm_pattern_classifier.detectors.dfs import DFSDetector


def test_bfs_detector_positives() -> None:
    """Test BFSDetector flags actual BFS implementations."""
    detector = BFSDetector()

    # Case 1: Standard BFS queue with collections.deque, popleft and append
    deque_bfs_code = (
        "def bfs_deque(graph, start):\n"
        "    visited = set()\n"
        "    queue = collections.deque([start])\n"
        "    while queue:\n"
        "        node = queue.popleft()\n"
        "        for neighbor in graph[node]:\n"
        "            if neighbor not in visited:\n"
        "                visited.add(neighbor)\n"
        "                queue.append(neighbor)\n"
    )
    result = detector.detect(ast.parse(deque_bfs_code))
    assert result is not None
    assert result.confidence == 0.90
    assert "bfs" in result.evidence[0].lower()

    # Case 2: BFS queue using standard list, pop(0) and extend (no visited check, confidence 0.80)
    list_bfs_code = (
        "def bfs_list(graph, start):\n"
        "    queue = [start]\n"
        "    while len(queue) > 0:\n"
        "        node = queue.pop(0)\n"
        "        queue.extend(graph[node])\n"
    )
    result2 = detector.detect(ast.parse(list_bfs_code))
    assert result2 is not None
    assert result2.confidence == 0.80


def test_bfs_detector_negatives() -> None:
    """Test BFSDetector does not flag near-misses or stack operations."""
    detector = BFSDetector()

    # Case 1: Stack operations (DF-like popping)
    stack_code = (
        "def stack_traverse(graph, start):\n"
        "    queue = [start]\n"
        "    while queue:\n"
        "        node = queue.pop()\n"
        "        queue.append(node)\n"
    )
    result = detector.detect(ast.parse(stack_code))
    assert result is None


def test_dfs_detector_positives() -> None:
    """Test DFSDetector flags actual DFS implementations."""
    detector = DFSDetector()

    # Case 1: Iterative DFS stack with pop and append
    iter_dfs_code = (
        "def dfs_iterative(graph, start):\n"
        "    visited = set()\n"
        "    stack = [start]\n"
        "    while stack:\n"
        "        node = stack.pop()\n"
        "        for neighbor in graph[node]:\n"
        "            if neighbor not in visited:\n"
        "                visited.add(neighbor)\n"
        "                stack.append(neighbor)\n"
    )
    result = detector.detect(ast.parse(iter_dfs_code))
    assert result is not None
    assert result.confidence == 0.90

    # Case 2: Recursive DFS
    recur_dfs_code = (
        "def dfs_recursive(graph, node, visited):\n"
        "    visited.add(node)\n"
        "    for neighbor in graph[node]:\n"
        "        if neighbor not in visited:\n"
        "            dfs_recursive(graph, neighbor, visited)\n"
    )
    result2 = detector.detect(ast.parse(recur_dfs_code))
    assert result2 is not None
    assert result2.confidence == 0.90


def test_dfs_detector_negatives() -> None:
    """Test DFSDetector does not flag queue operations or non-traversal recursions."""
    detector = DFSDetector()

    # Case 1: BFS implementation (should be detected as BFS, not DFS)
    bfs_code = (
        "def bfs_traverse(graph, start):\n"
        "    queue = [start]\n"
        "    while queue:\n"
        "        node = queue.pop(0)\n"
        "        queue.append(node)\n"
    )
    result = detector.detect(ast.parse(bfs_code))
    assert result is None

    # Case 2: Ordinary mathematical recursion (e.g. factorial)
    factorial_code = (
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n"
    )
    result2 = detector.detect(ast.parse(factorial_code))
    assert result2 is None

    # Case 3: Factorial with else branch (would previously trigger DFS detector)
    factorial_else_code = (
        "def factorial_else(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    else:\n"
        "        return n * factorial_else(n - 1)\n"
    )
    result3 = detector.detect(ast.parse(factorial_else_code))
    assert result3 is None

    # Case 4: Recursive Binary Search (ordinary binary search recursion)
    bsearch_code = (
        "def bsearch(arr, lo, hi, target):\n"
        "    if lo > hi:\n"
        "        return -1\n"
        "    mid = (lo + hi) // 2\n"
        "    if arr[mid] < target:\n"
        "        return bsearch(arr, mid + 1, hi, target)\n"
        "    return bsearch(arr, lo, mid - 1, target)\n"
    )
    result4 = detector.detect(ast.parse(bsearch_code))
    assert result4 is None

    # Case 5: Fibonacci with decorator memoization (DP, not DFS)
    fib_decorator = (
        "from functools import cache\n"
        "@cache\n"
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n-1) + fib(n-2)\n"
    )
    result5 = detector.detect(ast.parse(fib_decorator))
    assert result5 is None

    # Case 6: Fibonacci with manual memoization dict (DP, not DFS)
    fib_manual = (
        "memo = {}\n"
        "def fib(n):\n"
        "    if n in memo:\n"
        "        return memo[n]\n"
        "    if n < 2:\n"
        "        return n\n"
        "    memo[n] = fib(n-1) + fib(n-2)\n"
        "    return memo[n]\n"
    )
    result6 = detector.detect(ast.parse(fib_manual))
    assert result6 is None


def test_graph_detectors_module_level() -> None:
    """Test BFS and DFS detectors can identify patterns at the module level."""
    bfs_module_code = (
        "import collections\n"
        "queue = collections.deque([start])\n"
        "while queue:\n"
        "    node = queue.popleft()\n"
        "    queue.append(node)\n"
    )
    assert BFSDetector().detect(ast.parse(bfs_module_code)) is not None

    dfs_module_code = "stack = []\nwhile stack:\n    node = stack.pop()\n    stack.append(node)\n"
    assert DFSDetector().detect(ast.parse(dfs_module_code)) is not None


def test_graph_detectors_boolean_operators_in_while() -> None:
    """Test that while queue and not found condition is correctly handled."""
    bfs_and_cond_code = (
        "queue = [start]\n"
        "while queue and not found:\n"
        "    node = queue.pop(0)\n"
        "    queue.append(node)\n"
    )
    assert BFSDetector().detect(ast.parse(bfs_and_cond_code)) is not None


def test_dfs_detector_deque_stack() -> None:
    """Test that DFS can detect stacks initialized using collections.deque."""
    deque_stack_code = (
        "stack = collections.deque([start])\n"
        "while stack:\n"
        "    node = stack.pop()\n"
        "    stack.append(node)\n"
    )
    assert DFSDetector().detect(ast.parse(deque_stack_code)) is not None


def test_graph_detectors_async_context() -> None:
    """Test BFS and DFS detection inside async functions."""
    async_bfs_code = (
        "async def async_bfs(graph, start):\n"
        "    queue = [start]\n"
        "    while queue:\n"
        "        node = queue.pop(0)\n"
        "        queue.append(node)\n"
    )
    assert BFSDetector().detect(ast.parse(async_bfs_code)) is not None

    async_dfs_code = (
        "async def async_dfs(graph, start):\n"
        "    stack = []\n"
        "    while stack:\n"
        "        node = stack.pop()\n"
        "        stack.append(node)\n"
    )
    assert DFSDetector().detect(ast.parse(async_dfs_code)) is not None
