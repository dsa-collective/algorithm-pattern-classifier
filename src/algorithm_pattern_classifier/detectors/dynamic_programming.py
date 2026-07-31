import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class DynamicProgrammingDetector(BaseDetector):
    """Detector for the Dynamic Programming algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.DYNAMIC_PROGRAMMING

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Detect evidence of the Dynamic Programming pattern in parsed AST.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        evidence: list[str] = []
        max_confidence = 0.0

        # Helper to find if a node contains subtraction (e.g. i - 1)
        def contains_subtraction(node: ast.AST) -> bool:
            found = False

            class SubtractionFinder(ast.NodeVisitor):
                def visit_BinOp(self, bin_node: ast.BinOp) -> None:
                    nonlocal found
                    if isinstance(bin_node.op, ast.Sub):
                        found = True
                    self.generic_visit(bin_node)

            SubtractionFinder().visit(node)
            return found

        class DPVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_dp = False
                self.evidence: list[str] = []
                self.confidence = 0.0

            def visit_FunctionDef(self, func_node: ast.FunctionDef) -> None:
                func_name = func_node.name

                # Check for recursion
                is_recursive = False

                class RecursionFinder(ast.NodeVisitor):
                    def visit_Call(self, call_node: ast.Call) -> None:
                        nonlocal is_recursive
                        if isinstance(call_node.func, ast.Name) and call_node.func.id == func_name:
                            is_recursive = True
                        self.generic_visit(call_node)

                RecursionFinder().visit(func_node)

                if is_recursive:
                    # 1. Check for cache decorator
                    has_cache_decorator = False
                    for dec in func_node.decorator_list:
                        # e.g., @cache or @lru_cache, @functools.cache
                        if (isinstance(dec, ast.Name) and dec.id in ("cache", "lru_cache")) or (
                            isinstance(dec, ast.Attribute) and dec.attr in ("cache", "lru_cache")
                        ):
                            has_cache_decorator = True
                        # e.g., @lru_cache(maxsize=...)
                        elif isinstance(dec, ast.Call):
                            func_dec = dec.func
                            if (
                                isinstance(func_dec, ast.Name)
                                and func_dec.id in ("cache", "lru_cache")
                            ) or (
                                isinstance(func_dec, ast.Attribute)
                                and func_dec.attr in ("cache", "lru_cache")
                            ):
                                has_cache_decorator = True

                    if has_cache_decorator:
                        self.found_dp = True
                        self.confidence = 0.95
                        self.evidence.append(
                            f"Line {func_node.lineno}: found recursive function '{func_name}' "
                            f"with cache/lru_cache decorator."
                        )

                    # 2. Check for manual memoization:
                    # Look for containment checks (e.g. `if x in memo:`)
                    # or subscript checks (e.g. `if memo[x] != -1:`)
                    # and writing back (e.g. `memo[x] = ...`)
                    if not has_cache_decorator:
                        memo_containers: set[str] = set()
                        has_read_check = False
                        has_write_store = False

                        class MemoVisitor(ast.NodeVisitor):
                            def visit_Compare(self, comp_node: ast.Compare) -> None:
                                # e.g., x in memo
                                if len(comp_node.ops) == 1 and isinstance(
                                    comp_node.ops[0], (ast.In, ast.NotIn)
                                ):
                                    for comp in comp_node.comparators:
                                        if isinstance(comp, ast.Name):
                                            memo_containers.add(comp.id)
                                self.generic_visit(comp_node)

                            def visit_Subscript(self, sub_node: ast.Subscript) -> None:
                                # e.g., memo[x] check in a condition or expression
                                if isinstance(sub_node.value, ast.Name):
                                    memo_containers.add(sub_node.value.id)
                                self.generic_visit(sub_node)

                        # Check conditional checks
                        for body_stmt in func_node.body:
                            if isinstance(body_stmt, ast.If):
                                MemoVisitor().visit(body_stmt.test)
                                if memo_containers:
                                    has_read_check = True

                        # Check writing back to those same containers
                        class StoreVisitor(ast.NodeVisitor):
                            def visit_Assign(self, assign_node: ast.Assign) -> None:
                                for target in assign_node.targets:
                                    if (
                                        isinstance(target, ast.Subscript)
                                        and isinstance(target.value, ast.Name)
                                        and target.value.id in memo_containers
                                    ):
                                        nonlocal has_write_store
                                        has_write_store = True
                                self.generic_visit(assign_node)

                        StoreVisitor().visit(func_node)

                        if has_read_check and has_write_store:
                            self.found_dp = True
                            self.confidence = 0.95
                            self.evidence.append(
                                f"Line {func_node.lineno}: found recursive function '{func_name}' "
                                f"with manual cache lookups/stores."
                            )

                self.generic_visit(func_node)

            def visit_For(self, loop_node: ast.For) -> None:
                self.check_tabulation(loop_node)
                self.generic_visit(loop_node)

            def visit_While(self, loop_node: ast.While) -> None:
                self.check_tabulation(loop_node)
                self.generic_visit(loop_node)

            def check_tabulation(self, loop_node: ast.For | ast.While) -> None:
                assigned_tables: set[str] = set()

                # Find all variables assigned to via subscripting in the loop body
                class SubscriptAssignVisitor(ast.NodeVisitor):
                    def visit_Assign(self, assign_node: ast.Assign) -> None:
                        for target in assign_node.targets:
                            # e.g., dp[i] = ... or dp[i][j] = ...
                            curr = target
                            while isinstance(curr, ast.Subscript):
                                curr = curr.value
                            if isinstance(curr, ast.Name):
                                assigned_tables.add(curr.id)
                        self.generic_visit(assign_node)

                SubscriptAssignVisitor().visit(loop_node)

                # Look for reads from these tables using subtraction index expressions
                referenced_prior_state = False
                detected_table = None

                class SubscriptReadVisitor(ast.NodeVisitor):
                    def visit_Subscript(self, sub_node: ast.Subscript) -> None:
                        curr = sub_node.value
                        while isinstance(curr, ast.Subscript):
                            # For nested subscripts, also check the slice (index)
                            # of the outer/inner subscripts
                            if contains_subtraction(curr.slice):
                                nonlocal referenced_prior_state, detected_table
                                base = curr.value
                                while isinstance(base, ast.Subscript):
                                    base = base.value
                                if isinstance(base, ast.Name) and base.id in assigned_tables:
                                    referenced_prior_state = True
                                    detected_table = base.id
                            curr = curr.value

                        if (
                            isinstance(curr, ast.Name)
                            and curr.id in assigned_tables
                            and contains_subtraction(sub_node.slice)
                        ):
                            referenced_prior_state = True
                            detected_table = curr.id
                        self.generic_visit(sub_node)

                SubscriptReadVisitor().visit(loop_node)

                if referenced_prior_state and detected_table:
                    self.found_dp = True
                    self.confidence = 0.95
                    self.evidence.append(
                        f"Line {loop_node.lineno}: found iterative tabulation "
                        f"pattern reading prior entries from table '{detected_table}'."
                    )

        visitor = DPVisitor()
        visitor.visit(code_ast)

        if visitor.found_dp:
            max_confidence = visitor.confidence
            evidence = visitor.evidence
            return PatternMatch(
                pattern=AlgorithmPattern.DYNAMIC_PROGRAMMING,
                confidence=max_confidence,
                evidence=evidence,
            )

        return None
