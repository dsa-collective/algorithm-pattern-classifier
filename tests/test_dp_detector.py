import ast

from algorithm_pattern_classifier.detectors.dynamic_programming import DynamicProgrammingDetector


def test_dp_detector_memoized_positives() -> None:
    """Test memoized recursion implementations are flagged."""
    detector = DynamicProgrammingDetector()

    # Case 1: Fibonacci with decorator memoization
    fib_decorator = (
        "from functools import cache\n"
        "@cache\n"
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n-1) + fib(n-2)\n"
    )
    result = detector.detect(ast.parse(fib_decorator))
    assert result is not None
    assert result.confidence > 0.8
    assert "cache" in result.evidence[0] or "lru_cache" in result.evidence[0]

    # Case 2: Fibonacci with manual memoization dict
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
    result = detector.detect(ast.parse(fib_manual))
    assert result is not None
    assert result.confidence > 0.8
    assert "manual cache" in result.evidence[0]


def test_dp_detector_tabulation_positives() -> None:
    """Test iterative tabulation implementations are flagged."""
    detector = DynamicProgrammingDetector()

    # Case 1: Fibonacci tabulated (1D)
    fib_tab = (
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    dp = [0] * (n + 1)\n"
        "    dp[1] = 1\n"
        "    for i in range(2, n + 1):\n"
        "        dp[i] = dp[i-1] + dp[i-2]\n"
        "    return dp[n]\n"
    )
    result = detector.detect(ast.parse(fib_tab))
    assert result is not None
    assert result.confidence > 0.8
    assert "tabulation" in result.evidence[0]
    assert "dp" in result.evidence[0]

    # Case 2: 0/1 Knapsack tabulated (2D)
    knapsack_tab = (
        "def knapsack(W, wt, val, n):\n"
        "    dp = [[0 for x in range(W + 1)] for x in range(n + 1)]\n"
        "    for i in range(n + 1):\n"
        "        for w in range(W + 1):\n"
        "            if i == 0 or w == 0:\n"
        "                dp[i][w] = 0\n"
        "            elif wt[i-1] <= w:\n"
        "                dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w])\n"
        "            else:\n"
        "                dp[i][w] = dp[i-1][w]\n"
        "    return dp[n][W]\n"
    )
    result = detector.detect(ast.parse(knapsack_tab))
    assert result is not None
    assert result.confidence > 0.8
    assert "tabulation" in result.evidence[0]
    assert "dp" in result.evidence[0]

    # Case 3: Longest Common Subsequence tabulated (2D)
    lcs_tab = (
        "def lcs(X, Y):\n"
        "    m = len(X)\n"
        "    n = len(Y)\n"
        "    L = [[0]*(n+1) for i in range(m+1)]\n"
        "    for i in range(m+1):\n"
        "        for j in range(n+1):\n"
        "            if i == 0 or j == 0:\n"
        "                L[i][j] = 0\n"
        "            elif X[i-1] == Y[j-1]:\n"
        "                L[i][j] = L[i-1][j-1] + 1\n"
        "            else:\n"
        "                L[i][j] = max(L[i-1][j], L[i][j-1])\n"
        "    return L[m][n]\n"
    )
    result = detector.detect(ast.parse(lcs_tab))
    assert result is not None
    assert result.confidence > 0.8
    assert "tabulation" in result.evidence[0]
    assert "L" in result.evidence[0]


def test_dp_detector_negatives() -> None:
    """Test DP detector does not flag plain loops or plain recursive functions."""
    detector = DynamicProgrammingDetector()

    # Case 1: Plain (non-memoized) recursion
    plain_recursion = "def recurse(n):\n    if n <= 1:\n        return n\n    return recurse(n-1)\n"
    result = detector.detect(ast.parse(plain_recursion))
    assert result is None

    # Case 2: Simple loop with no prior table entry lookup
    simple_loop = (
        "def simple(arr):\n"
        "    out = [0] * len(arr)\n"
        "    for i in range(len(arr)):\n"
        "        out[i] = arr[i] * 2\n"
        "    return out\n"
    )
    result = detector.detect(ast.parse(simple_loop))
    assert result is None
