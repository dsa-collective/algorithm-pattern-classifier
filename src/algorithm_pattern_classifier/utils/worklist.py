import ast


def is_worklist_init(value: ast.expr) -> bool:
    """Check if the AST expression represents a worklist (queue/stack) initialization.

    Note: Treating all list literals as candidate worklists is a broad heuristic
    suited for a static classifier; whether the list functions as a FIFO queue
    or LIFO stack is determined by the loop traversal operations (e.g., pop(0) vs pop()).
    """
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name) and func.id == "deque":
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "deque"
            and isinstance(func.value, ast.Name)
            and func.value.id == "collections"
        ):
            return True
    return isinstance(value, ast.List)


def get_loop_test_variables(test: ast.expr) -> set[str]:
    """Extract all candidate worklist variable names from a loop test expression.

    Recursively resolves logical operators (BoolOp, UnaryOp) and length comparisons
    to find all variable names participating in the loop condition.
    """
    vars_found = set()

    if isinstance(test, ast.Name):
        vars_found.add(test.id)
    elif isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == "len":
        if len(test.args) == 1 and isinstance(test.args[0], ast.Name):
            vars_found.add(test.args[0].id)
    elif isinstance(test, ast.Compare):
        vars_found.update(get_loop_test_variables(test.left))
        for comp in test.comparators:
            vars_found.update(get_loop_test_variables(comp))
    elif isinstance(test, ast.BoolOp):
        for val in test.values:
            vars_found.update(get_loop_test_variables(val))
    elif isinstance(test, ast.UnaryOp):
        vars_found.update(get_loop_test_variables(test.operand))

    return vars_found
