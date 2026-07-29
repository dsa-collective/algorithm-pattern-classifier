import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch
from algorithm_pattern_classifier.utils.worklist import get_loop_test_variables, is_worklist_init


class BFSDetector(BaseDetector):
    """Detector for the Breadth-First Search (BFS) algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.BFS

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Parse AST and detect BFS pattern.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        evidence: list[str] = []
        best_confidence = 0.0

        def collect_initializations(body: list[ast.stmt]) -> dict[str, ast.AST]:
            inits = {}

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

        class BFSVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_bfs = False
                self.evidence: list[str] = []
                self.confidence = 0.0
                self.initialized_queues: dict[str, ast.AST] = {}

            def visit_Module(self, node: ast.Module) -> None:
                old_queues = self.initialized_queues.copy()
                self.initialized_queues.update(collect_initializations(node.body))
                self.generic_visit(node)
                self.initialized_queues = old_queues

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                old_queues = self.initialized_queues.copy()
                self.initialized_queues = collect_initializations(node.body)
                self.generic_visit(node)
                self.initialized_queues = old_queues

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                old_queues = self.initialized_queues.copy()
                self.initialized_queues = collect_initializations(node.body)
                self.generic_visit(node)
                self.initialized_queues = old_queues

            def visit_While(self, node: ast.While) -> None:
                loop_vars = get_loop_test_variables(node.test)
                for queue_var in loop_vars:
                    if queue_var in self.initialized_queues:

                        class LoopBodyVisitor(ast.NodeVisitor):
                            def __init__(self, q_var: str) -> None:
                                self.queue_var = q_var
                                self.has_pop_front = False
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
                                    and func.value.id == self.queue_var
                                ):
                                    if func.attr == "popleft":
                                        self.has_pop_front = True
                                        self.pop_line = child.lineno
                                    elif func.attr == "pop" and len(child.args) == 1:
                                        arg = child.args[0]
                                        if isinstance(arg, ast.Constant) and arg.value == 0:
                                            self.has_pop_front = True
                                            self.pop_line = child.lineno
                                    elif func.attr in ("append", "extend"):
                                        self.has_append = True
                                        self.append_line = child.lineno
                                self.generic_visit(child)

                        body_visitor = LoopBodyVisitor(queue_var)
                        for stmt in node.body:
                            body_visitor.visit(stmt)

                        if body_visitor.has_pop_front and body_visitor.has_append:
                            self.found_bfs = True

                            # Graded confidence
                            has_visited_check = False
                            for sub in ast.walk(node):
                                if isinstance(sub, ast.Name) and any(
                                    kw in sub.id.lower() for kw in ("visited", "seen")
                                ):
                                    has_visited_check = True
                                    break

                            confidence = 0.85
                            if has_visited_check:
                                confidence += 0.05
                            confidence += 0.05  # Pop/append on the same queue variable

                            self.confidence = max(self.confidence, min(0.95, confidence))
                            init_line = getattr(self.initialized_queues[queue_var], "lineno", 0)
                            self.evidence.append(
                                f"Line {node.lineno}: Found BFS graph traversal pattern "
                                f"using queue '{queue_var}' initialized at line {init_line}. "
                                f"Front pop at line {body_visitor.pop_line}, node "
                                f"expansion at line {body_visitor.append_line}."
                            )

                self.generic_visit(node)

        visitor = BFSVisitor()
        visitor.visit(code_ast)

        if visitor.found_bfs:
            best_confidence = visitor.confidence
            evidence = visitor.evidence
            return PatternMatch(
                pattern=AlgorithmPattern.BFS,
                confidence=round(best_confidence, 2),
                evidence=evidence,
            )

        return None
