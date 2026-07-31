import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch
from algorithm_pattern_classifier.utils.worklist import get_loop_test_variables, is_worklist_init


class DFSDetector(BaseDetector):
    """Detector for the Depth-First Search (DFS) algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.DFS

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Parse AST and detect DFS pattern.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        evidence: list[str] = []
        best_confidence = 0.0

        def collect_initializations(body: list[ast.stmt]) -> dict[str, ast.AST]:
            inits: dict[str, ast.AST] = {}

            class InitCollector(ast.NodeVisitor):
                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    pass

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                    pass

                def visit_ClassDef(self, node: ast.ClassDef) -> None:
                    pass

                def visit_Assign(self, node: ast.Assign) -> None:
                    for target in node.targets:
                        if isinstance(target, ast.Name) and is_worklist_init(node.value):
                            inits[target.id] = node
                    self.generic_visit(node)

                def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                    if (
                        isinstance(node.target, ast.Name)
                        and node.value
                        and is_worklist_init(node.value)
                    ):
                        inits[node.target.id] = node
                    self.generic_visit(node)

            collector = InitCollector()
            for stmt in body:
                collector.visit(stmt)
            return inits

        class DFSVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_dfs = False
                self.evidence: list[str] = []
                self.confidence = 0.0
                self.initialized_stacks: dict[str, ast.AST] = {}

            def visit_Module(self, node: ast.Module) -> None:
                old_stacks = self.initialized_stacks.copy()
                self.initialized_stacks.update(collect_initializations(node.body))
                self.generic_visit(node)
                self.initialized_stacks = old_stacks

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                old_stacks = self.initialized_stacks.copy()
                self.initialized_stacks = collect_initializations(node.body)
                self._check_recursive_dfs(node)
                self.generic_visit(node)
                self.initialized_stacks = old_stacks

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                old_stacks = self.initialized_stacks.copy()
                self.initialized_stacks = collect_initializations(node.body)
                self._check_recursive_dfs(node)
                self.generic_visit(node)
                self.initialized_stacks = old_stacks

            def visit_While(self, node: ast.While) -> None:
                loop_vars = get_loop_test_variables(node.test)
                for stack_var in loop_vars:
                    if stack_var in self.initialized_stacks:

                        class LoopBodyVisitor(ast.NodeVisitor):
                            def __init__(self, s_var: str) -> None:
                                self.stack_var = s_var
                                self.has_pop_back = False
                                self.has_append = False
                                self.pop_line = 0
                                self.append_line = 0

                            def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
                                pass

                            def visit_AsyncFunctionDef(self, n: ast.AsyncFunctionDef) -> None:
                                pass

                            def visit_ClassDef(self, n: ast.ClassDef) -> None:
                                pass

                            def visit_Call(self, child: ast.Call) -> None:
                                func = child.func
                                if (
                                    isinstance(func, ast.Attribute)
                                    and isinstance(func.value, ast.Name)
                                    and func.value.id == self.stack_var
                                ):
                                    if func.attr == "pop":
                                        if len(child.args) == 0:
                                            self.has_pop_back = True
                                            self.pop_line = getattr(child, "lineno", 0)
                                        elif len(child.args) == 1:
                                            arg = child.args[0]
                                            if (
                                                isinstance(arg, ast.UnaryOp)
                                                and isinstance(arg.op, ast.USub)
                                                and isinstance(arg.operand, ast.Constant)
                                                and arg.operand.value == 1
                                            ):
                                                self.has_pop_back = True
                                                self.pop_line = getattr(child, "lineno", 0)
                                    elif func.attr in ("append", "extend"):
                                        self.has_append = True
                                        self.append_line = child.lineno
                                self.generic_visit(child)

                        body_visitor = LoopBodyVisitor(stack_var)
                        for stmt in node.body:
                            body_visitor.visit(stmt)

                        if body_visitor.has_pop_back and body_visitor.has_append:
                            self.found_dfs = True

                            # Graded confidence
                            has_visited_check = False
                            for sub in ast.walk(node):
                                if isinstance(sub, ast.Name) and any(
                                    kw in sub.id.lower() for kw in ("visited", "seen")
                                ):
                                    has_visited_check = True
                                    break

                            confidence = 0.80
                            if has_visited_check:
                                confidence += 0.10

                            self.confidence = max(self.confidence, min(0.95, confidence))
                            init_line = getattr(self.initialized_stacks[stack_var], "lineno", 0)
                            self.evidence.append(
                                f"Line {node.lineno}: Found iterative DFS graph traversal pattern "
                                f"using stack '{stack_var}' initialized at line {init_line}. "
                                f"End pop at line {body_visitor.pop_line}, node "
                                f"expansion at line {body_visitor.append_line}."
                            )

                self.generic_visit(node)

            def _check_recursive_dfs(
                self, func_node: ast.FunctionDef | ast.AsyncFunctionDef
            ) -> None:
                func_name = func_node.name

                def has_recursive_call(val_node: ast.AST) -> bool:
                    for sub in ast.walk(val_node):
                        if isinstance(sub, ast.Call) and (
                            (isinstance(sub.func, ast.Name) and sub.func.id == func_name)
                            or (
                                isinstance(sub.func, ast.Attribute)
                                and isinstance(sub.func.value, ast.Name)
                                and sub.func.value.id == "self"
                                and sub.func.attr == func_name
                            )
                        ):
                            return True
                    return False

                class RecursiveDFSHeuristicVisitor(ast.NodeVisitor):
                    def __init__(self) -> None:
                        self.found_dfs = False
                        self.in_for_loop = False
                        self.has_visited_or_seen = False
                        self.has_grid_marking = False
                        self.recursive_calls_in_for: list[ast.Call] = []
                        self.all_recursive_calls: list[ast.Call] = []

                    def visit_Name(self, node: ast.Name) -> None:
                        name_lower = node.id.lower()
                        if "visited" in name_lower or "seen" in name_lower:
                            self.has_visited_or_seen = True
                        self.generic_visit(node)

                    def visit_Assign(self, node: ast.Assign) -> None:
                        for target in node.targets:
                            if isinstance(target, ast.Subscript) and not has_recursive_call(
                                node.value
                            ):
                                self.has_grid_marking = True
                        self.generic_visit(node)

                    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                        if (
                            isinstance(node.target, ast.Subscript)
                            and node.value
                            and not has_recursive_call(node.value)
                        ):
                            self.has_grid_marking = True
                        self.generic_visit(node)

                    def visit_AugAssign(self, node: ast.AugAssign) -> None:
                        if isinstance(node.target, ast.Subscript) and not has_recursive_call(
                            node.value
                        ):
                            self.has_grid_marking = True
                        self.generic_visit(node)

                    def visit_For(self, node: ast.For) -> None:
                        old_in_for = self.in_for_loop
                        self.in_for_loop = True
                        self.generic_visit(node)
                        self.in_for_loop = old_in_for

                    def visit_Call(self, node: ast.Call) -> None:
                        is_recur = False
                        if (isinstance(node.func, ast.Name) and node.func.id == func_name) or (
                            isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == "self"
                            and node.func.attr == func_name
                        ):
                            is_recur = True

                        if is_recur:
                            self.all_recursive_calls.append(node)
                            if self.in_for_loop:
                                self.recursive_calls_in_for.append(node)
                        self.generic_visit(node)

                visitor = RecursiveDFSHeuristicVisitor()
                visitor.visit(func_node)

                if visitor.all_recursive_calls:
                    # Traversal check: recursive call inside For loop (e.g. neighbors traversal)
                    # OR presence of visited/seen tracking variable or grid modification.
                    is_dfs = (
                        bool(visitor.recursive_calls_in_for)
                        or visitor.has_visited_or_seen
                        or visitor.has_grid_marking
                    )

                    if is_dfs:
                        self.found_dfs = True
                        confidence = 0.80
                        if visitor.has_visited_or_seen:
                            confidence += 0.05
                        if visitor.recursive_calls_in_for:
                            confidence += 0.05

                        self.confidence = max(self.confidence, min(0.95, confidence))
                        call_line = visitor.all_recursive_calls[0].lineno
                        self.evidence.append(
                            f"Line {func_node.lineno}: Found recursive DFS pattern in "
                            f"function '{func_name}' with recursive call at line "
                            f"{call_line}."
                        )

        visitor = DFSVisitor()
        visitor.visit(code_ast)

        if visitor.found_dfs:
            best_confidence = visitor.confidence
            evidence = visitor.evidence
            return PatternMatch(
                pattern=AlgorithmPattern.DFS,
                confidence=round(best_confidence, 2),
                evidence=evidence,
            )

        return None
