import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class SlidingWindowDetector(BaseDetector):
    """Detector for the Sliding Window algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.SLIDING_WINDOW

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Detect evidence of the Sliding Window pattern in parsed AST.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        evidence: list[str] = []
        max_confidence = 0.0

        class SlidingWindowVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found_sliding_window = False
                self.evidence: list[str] = []
                self.confidence = 0.0

            def visit_For(self, node: ast.For) -> None:
                # 1. Loop variable (end pointer)
                if not isinstance(node.target, ast.Name):
                    self.generic_visit(node)
                    return

                end_var = node.target.id

                # 2. Identify variables updated in the loop (candidate start pointers)
                updated_vars: set[str] = set()

                class UpdateVisitor(ast.NodeVisitor):
                    def visit_Assign(self, assign_node: ast.Assign) -> None:
                        for target in assign_node.targets:
                            if isinstance(target, ast.Name):
                                updated_vars.add(target.id)
                        self.generic_visit(assign_node)

                    def visit_AugAssign(self, aug_node: ast.AugAssign) -> None:
                        if isinstance(aug_node.target, ast.Name):
                            updated_vars.add(aug_node.target.id)
                        self.generic_visit(aug_node)

                UpdateVisitor().visit(node)
                updated_vars.discard(end_var)

                # 3. Check for window size calculation (end_var - start_var)
                # or common sequence subscripting by both end_var and start_var
                has_window_metric = False
                detected_start_var = None

                class MetricVisitor(ast.NodeVisitor):
                    def __init__(self) -> None:
                        self.start_vars_in_subtraction: set[str] = set()
                        self.subscript_vars: dict[str, set[str]] = {}

                    def visit_BinOp(self, bin_node: ast.BinOp) -> None:
                        if isinstance(bin_node.op, ast.Sub):
                            left_id = (
                                bin_node.left.id if isinstance(bin_node.left, ast.Name) else None
                            )
                            right_id = (
                                bin_node.right.id if isinstance(bin_node.right, ast.Name) else None
                            )
                            if left_id == end_var and right_id in updated_vars:
                                assert right_id is not None
                                self.start_vars_in_subtraction.add(right_id)
                            elif right_id == end_var and left_id in updated_vars:
                                assert left_id is not None
                                self.start_vars_in_subtraction.add(left_id)
                        self.generic_visit(bin_node)

                    def visit_Subscript(self, sub_node: ast.Subscript) -> None:
                        # Handle slicing like arr[start:end]
                        if isinstance(sub_node.slice, ast.Slice):
                            lower_id = (
                                sub_node.slice.lower.id
                                if isinstance(sub_node.slice.lower, ast.Name)
                                else None
                            )
                            upper_id = (
                                sub_node.slice.upper.id
                                if isinstance(sub_node.slice.upper, ast.Name)
                                else None
                            )
                            if lower_id in updated_vars and upper_id == end_var:
                                assert lower_id is not None
                                self.start_vars_in_subtraction.add(lower_id)
                        # Track indexing access to detect joint subscripting
                        elif isinstance(sub_node.value, ast.Name):
                            seq_name = sub_node.value.id
                            if isinstance(sub_node.slice, ast.Name):
                                self.subscript_vars.setdefault(seq_name, set()).add(
                                    sub_node.slice.id
                                )
                            elif (
                                isinstance(sub_node.slice, ast.BinOp)
                                and isinstance(sub_node.slice.op, ast.Sub)
                                and isinstance(sub_node.slice.left, ast.Name)
                                and sub_node.slice.left.id == end_var
                            ):
                                right = sub_node.slice.right
                                if isinstance(right, ast.Name):
                                    self.subscript_vars.setdefault(seq_name, set()).add(
                                        f"sub-{right.id}"
                                    )
                                elif isinstance(right, ast.Constant) and isinstance(
                                    right.value, (int, str)
                                ):
                                    self.subscript_vars.setdefault(seq_name, set()).add(
                                        f"sub-{right.value}"
                                    )
                        self.generic_visit(sub_node)

                metric_visitor = MetricVisitor()
                metric_visitor.visit(node)

                # Check subtraction criteria
                if metric_visitor.start_vars_in_subtraction:
                    has_window_metric = True
                    detected_start_var = next(iter(metric_visitor.start_vars_in_subtraction))

                # Check joint subscripting criteria
                if not has_window_metric:
                    for _seq, indices in metric_visitor.subscript_vars.items():
                        if end_var in indices:
                            # Check dynamic start variables (variables updated in the loop)
                            common_starts = indices.intersection(updated_vars)
                            if common_starts:
                                has_window_metric = True
                                detected_start_var = next(iter(common_starts))
                                break
                            # Check fixed window size subtraction elements (e.g. sub-k or sub-3)
                            sub_starts = [idx for idx in indices if idx.startswith("sub-")]
                            if sub_starts:
                                has_window_metric = True
                                # Extract variable name / constant value from prefix
                                detected_start_var = sub_starts[0].split("-", 1)[1]
                                break

                if has_window_metric and detected_start_var:
                    self.found_sliding_window = True
                    self.confidence = 0.95
                    self.evidence.append(
                        f"Line {node.lineno}: found sliding window loop with "
                        f"expansion pointer '{end_var}' and boundary offset/pointer "
                        f"'{detected_start_var}'."
                    )

                self.generic_visit(node)

        visitor = SlidingWindowVisitor()
        visitor.visit(code_ast)

        if visitor.found_sliding_window:
            max_confidence = visitor.confidence
            evidence = visitor.evidence
            return PatternMatch(
                pattern=AlgorithmPattern.SLIDING_WINDOW,
                confidence=max_confidence,
                evidence=evidence,
            )

        return None
