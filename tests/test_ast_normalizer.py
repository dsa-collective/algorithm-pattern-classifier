import ast

from algorithm_pattern_classifier.classifiers.pattern_classifier import PatternClassifier
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern
from algorithm_pattern_classifier.utils.ast_normalizer import ASTNormalizer


def test_ast_normalizer_tuple_unpacking() -> None:
    """Test ASTNormalizer desugars tuple unpacking assignments."""
    code = "left, right = 0, len(arr) - 1"
    tree = ast.parse(code)
    normalized = ASTNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)

    # Should transform to separate simple assignments for left and right
    assert isinstance(normalized, ast.Module)
    assert len(normalized.body) == 2
    assert isinstance(normalized.body[0], ast.Assign)
    assert isinstance(normalized.body[0].targets[0], ast.Name)
    # variables are canonicalized
    assert normalized.body[0].targets[0].id == "var_0"
    assert isinstance(normalized.body[1], ast.Assign)
    assert isinstance(normalized.body[1].targets[0], ast.Name)
    assert normalized.body[1].targets[0].id == "var_1"


def test_ast_normalizer_non_tuple_unpacking() -> None:
    """Test ASTNormalizer leaves function-matched unpacking as-is."""
    code = "left, right = get_bounds()"
    tree = ast.parse(code)
    normalized = ASTNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)

    assert isinstance(normalized, ast.Module)
    assert len(normalized.body) == 1
    assert isinstance(normalized.body[0], ast.Assign)
    assert isinstance(normalized.body[0].targets[0], ast.Tuple)


def test_two_pointers_unpacking_integration() -> None:
    """Test that a function using tuple unpacking is recognized correctly."""
    code = (
        "def two_sum(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left < right:\n"
        "        val = arr[left] + arr[right]\n"
        "        if val == target:\n"
        "            return True\n"
        "        if val < target:\n"
        "            left += 1\n"
        "        else:\n"
        "            right -= 1\n"
        "    return False\n"
    )
    classifier = PatternClassifier()
    results = classifier.classify(code)

    assert len(results) > 0
    assert results[0].pattern == AlgorithmPattern.TWO_POINTERS
    assert results[0].confidence >= 0.8


def test_ast_normalizer_starred_unpacking() -> None:
    """Test ASTNormalizer leaves starred unpacking assignments as-is."""
    code = "a, *b = 1, 2"
    tree = ast.parse(code)
    normalized = ASTNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)

    assert isinstance(normalized, ast.Module)
    assert len(normalized.body) == 1
    assert isinstance(normalized.body[0], ast.Assign)
    assert isinstance(normalized.body[0].targets[0], ast.Tuple)


def test_ast_normalizer_swap_unchanged() -> None:
    """Test ASTNormalizer preserves variable swap assignments without desugaring."""
    code = "a, b = b, a"
    tree = ast.parse(code)
    normalized = ASTNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)

    assert isinstance(normalized, ast.Module)
    assert len(normalized.body) == 1
    assert isinstance(normalized.body[0], ast.Assign)
    assert isinstance(normalized.body[0].targets[0], ast.Tuple)


def test_ast_normalizer_subscript_swap_unchanged() -> None:
    """Test ASTNormalizer preserves subscript array swap assignments without desugaring."""
    code = "arr[i], arr[j] = arr[j], arr[i]"
    tree = ast.parse(code)
    normalized = ASTNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)

    assert isinstance(normalized, ast.Module)
    assert len(normalized.body) == 1
    assert isinstance(normalized.body[0], ast.Assign)
    assert isinstance(normalized.body[0].targets[0], ast.Tuple)


def test_two_pointers_swap_integration() -> None:
    """Test that a function using pointer swapping is recognized correctly as two pointers."""
    code = (
        "def reverse_array(arr):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left < right:\n"
        "        arr[left], arr[right] = arr[right], arr[left]\n"
        "        left += 1\n"
        "        right -= 1\n"
        "    return arr\n"
    )
    classifier = PatternClassifier()
    results = classifier.classify(code)

    assert len(results) > 0
    assert results[0].pattern == AlgorithmPattern.TWO_POINTERS
    assert results[0].confidence >= 0.8


def test_ast_normalizer_canonical_renaming_identical_structure() -> None:
    """Test identical algorithms with different variable names produce identical ASTs."""
    code1 = """
def reverse_array(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr
"""
    code2 = """
def backwards_array(a):
    i, j = 0, len(a) - 1
    while i < j:
        a[i], a[j] = a[j], a[i]
        i += 1
        j -= 1
    return a
"""
    tree1 = ASTNormalizer().visit(ast.parse(code1))
    tree2 = ASTNormalizer().visit(ast.parse(code2))

    assert ast.dump(tree1) == ast.dump(tree2)
