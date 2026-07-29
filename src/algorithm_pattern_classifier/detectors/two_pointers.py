import ast

from algorithm_pattern_classifier.interfaces.detector import BaseDetector
from algorithm_pattern_classifier.models.patterns import AlgorithmPattern, PatternMatch


class TwoPointersDetector(BaseDetector):
    """Detector for the Two Pointers algorithmic design pattern."""

    @property
    def pattern(self) -> AlgorithmPattern:
        return AlgorithmPattern.TWO_POINTERS

    def detect(self, code_ast: ast.AST) -> PatternMatch | None:
        """Parse AST and detect Two Pointers pattern.

        Args:
            code_ast: The parsed AST of the source code.

        Returns:
            A PatternMatch representing the detection outcome, or None.
        """
        best_confidence = 0.0
        best_evidence: list[str] = []

        class TwoPointersVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.parent_stack: list[ast.AST] = []
                self.statements_stack: list[list[ast.stmt]] = []

            def visit_Module(self, node: ast.Module) -> None:
                self.parent_stack.append(node)
                self.statements_stack.append(node.body)
                self.generic_visit(node)
                self.statements_stack.pop()
                self.parent_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.parent_stack.append(node)
                self.statements_stack.append(node.body)
                self.generic_visit(node)
                self.statements_stack.pop()
                self.parent_stack.pop()

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.parent_stack.append(node)
                self.statements_stack.append(node.body)
                self.generic_visit(node)
                self.statements_stack.pop()
                self.parent_stack.pop()

            def visit_While(self, node: ast.While) -> None:
                nonlocal best_confidence, best_evidence

                # Check if loop condition compares two variables
                if (
                    isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and len(node.test.comparators) == 1
                    and isinstance(node.test.comparators[0], ast.Name)
                ):
                    var1 = node.test.left.id
                    var2 = node.test.comparators[0].id
                    op = node.test.ops[0]

                    # Loop condition checks left < right (or <=, >, >=, !=)
                    if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.NotEq)):
                        loop_confidence = 0.4
                        loop_evidence = [
                            f"Line {node.lineno}: Loop condition compares variables "
                            f"'{var1}' and '{var2}'."
                        ]

                        # Check Rule 1: Both pointers initialized near each other (+0.3)
                        # We look at preceding statements in the current block
                        if self.statements_stack:
                            current_body = self.statements_stack[-1]
                            try:
                                loop_idx = current_body.index(node)
                                preceding = current_body[:loop_idx]
                            except ValueError:
                                preceding = []

                            assigned_vars = self._get_assigned_vars(preceding)

                            # Also check function arguments if the parent is a function
                            if self.parent_stack:
                                parent = self.parent_stack[-1]
                                if isinstance(parent, ast.FunctionDef):
                                    for arg in parent.args.args:
                                        assigned_vars.add(arg.arg)

                            if var1 in assigned_vars and var2 in assigned_vars:
                                loop_confidence += 0.3
                                loop_evidence.append(
                                    f"Line {node.lineno}: Both pointer variables "
                                    f"'{var1}' and '{var2}' are initialized before the loop."
                                )

                        # Check Rule 3: Pointers incremented/decremented inside the loop (+0.3)
                        var1_updated = False
                        var2_updated = False

                        for child in ast.walk(node):
                            if self._is_pointer_update(child, var1):
                                var1_updated = True
                            if self._is_pointer_update(child, var2):
                                var2_updated = True

                        if var1_updated and var2_updated:
                            loop_confidence += 0.3
                            loop_evidence.append(
                                f"Line {node.lineno}: Both pointer variables "
                                f"'{var1}' and '{var2}' are updated inside the loop."
                            )
                        else:
                            loop_confidence = 0.0

                        if loop_confidence > best_confidence:
                            best_confidence = loop_confidence
                            best_evidence = loop_evidence

                self.parent_stack.append(node)
                self.statements_stack.append(node.body)
                self.generic_visit(node)
                self.statements_stack.pop()
                self.parent_stack.pop()

            def _get_assigned_vars(self, statements: list[ast.stmt]) -> set[str]:
                assigned: set[str] = set()
                for stmt in statements:
                    for sub_node in ast.walk(stmt):
                        if isinstance(sub_node, ast.Assign):
                            for target in sub_node.targets:
                                if isinstance(target, ast.Name):
                                    assigned.add(target.id)
                                elif isinstance(target, (ast.Tuple, ast.List)):
                                    for elt in target.elts:
                                        if isinstance(elt, ast.Name):
                                            assigned.add(elt.id)
                        elif isinstance(sub_node, ast.AnnAssign) and isinstance(
                            sub_node.target, ast.Name
                        ):
                            assigned.add(sub_node.target.id)
                return assigned

            def _is_pointer_update(self, node: ast.AST, var_name: str) -> bool:
                if (
                    isinstance(node, ast.AugAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == var_name
                    and isinstance(node.op, (ast.Add, ast.Sub))
                ):
                    return True
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == var_name
                            and isinstance(node.value, ast.BinOp)
                            and isinstance(node.value.op, (ast.Add, ast.Sub))
                        ):
                            left_is_var = (
                                isinstance(node.value.left, ast.Name)
                                and node.value.left.id == var_name
                            )
                            right_is_var = (
                                isinstance(node.value.right, ast.Name)
                                and node.value.right.id == var_name
                            )
                            if left_is_var or right_is_var:
                                return True
                        elif (
                            isinstance(target, (ast.Tuple, ast.List))
                            and isinstance(node.value, (ast.Tuple, ast.List))
                            and len(target.elts) == len(node.value.elts)
                        ):
                            for i, t_elt in enumerate(target.elts):
                                if isinstance(t_elt, ast.Name) and t_elt.id == var_name:
                                    val_elt = node.value.elts[i]
                                    if isinstance(val_elt, ast.BinOp) and isinstance(
                                        val_elt.op, (ast.Add, ast.Sub)
                                    ):
                                        left_is_var = (
                                            isinstance(val_elt.left, ast.Name)
                                            and val_elt.left.id == var_name
                                        )
                                        right_is_var = (
                                            isinstance(val_elt.right, ast.Name)
                                            and val_elt.right.id == var_name
                                        )
                                        if left_is_var or right_is_var:
                                            return True
                return False

        visitor = TwoPointersVisitor()
        visitor.visit(code_ast)

        if best_confidence > 0.0:
            return PatternMatch(
                pattern=AlgorithmPattern.TWO_POINTERS,
                confidence=round(best_confidence, 2),
                evidence=best_evidence,
            )

        return None
