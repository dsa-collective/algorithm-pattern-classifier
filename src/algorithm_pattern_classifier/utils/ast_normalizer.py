import ast


class ASTNormalizer(ast.NodeTransformer):
    """Pre-processes and normalizes Python AST structures before classification.

    This simplifies complex syntaxes (e.g. tuple unpacking assignments) to make
    pattern detection more robust, and applies canonical variable renaming
    so that mathematically identical algorithms hash identically regardless of
    the original variable names used.
    """

    def __init__(self):
        super().__init__()
        self.scopes = [{}]
        self.var_counter = 0

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        self.scopes.pop()

    def declare_var(self, name: str) -> str:
        if name in self.scopes[-1]:
            return self.scopes[-1][name]
        canonical_name = f"var_{self.var_counter}"
        self.var_counter += 1
        self.scopes[-1][name] = canonical_name
        return canonical_name

    def lookup_var(self, name: str) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return name

    def _visit_list(self, nodes: list[ast.AST]) -> list[ast.AST]:
        result = []
        for n in nodes:
            res = self.visit(n)
            if isinstance(res, list):
                result.extend(res)
            elif res is not None:
                result.append(res)
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.decorator_list = self._visit_list(node.decorator_list)
        if node.returns:
            node.returns = self.visit(node.returns)

        node.name = self.declare_var(node.name)

        self.push_scope()
        node.args = self.visit(node.args)
        node.body = self._visit_list(node.body)
        self.pop_scope()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.decorator_list = self._visit_list(node.decorator_list)
        if node.returns:
            node.returns = self.visit(node.returns)

        node.name = self.declare_var(node.name)

        self.push_scope()
        node.args = self.visit(node.args)
        node.body = self._visit_list(node.body)
        self.pop_scope()
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.decorator_list = self._visit_list(node.decorator_list)
        node.bases = self._visit_list(node.bases)
        node.keywords = self._visit_list(node.keywords)

        node.name = self.declare_var(node.name)

        self.push_scope()
        node.body = self._visit_list(node.body)
        self.pop_scope()
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        self.push_scope()
        node.args = self.visit(node.args)
        node.body = self.visit(node.body)
        self.pop_scope()
        return node

    def _visit_comprehension_node(self, node: ast.AST) -> ast.AST:
        self.push_scope()
        self.generic_visit(node)
        self.pop_scope()
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._visit_comprehension_node(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._visit_comprehension_node(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self._visit_comprehension_node(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._visit_comprehension_node(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.annotation:
            node.annotation = self.visit(node.annotation)
        node.arg = self.declare_var(node.arg)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Store):
            node.id = self.declare_var(node.id)
        elif isinstance(node.ctx, (ast.Load, ast.Del)):
            node.id = self.lookup_var(node.id)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.AST]:
        """Normalize assignments by desugaring tuple unpackings if possible.

        Args:
            node: The Assign node to normalize.

        Returns:
            The normalized AST node or list of nodes.
        """
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Tuple)
            and len(node.targets[0].elts) == len(node.value.elts)
            and not any(isinstance(elt, ast.Starred) for elt in node.targets[0].elts)
        ):
            target_names = {n.id for n in ast.walk(node.targets[0]) if isinstance(n, ast.Name)}
            value_names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}

            if target_names & value_names:
                return self.generic_visit(node)

            new_nodes: list[ast.AST] = []
            for target, val in zip(node.targets[0].elts, node.value.elts, strict=True):
                new_assign = ast.Assign(targets=[target], value=val)
                ast.copy_location(new_assign, node)
                new_nodes.append(new_assign)

            resolved_nodes: list[ast.AST] = []
            for n in new_nodes:
                visited = self.visit(n)
                if isinstance(visited, list):
                    resolved_nodes.extend(visited)
                elif visited is not None:
                    resolved_nodes.append(visited)
            return resolved_nodes

        return self.generic_visit(node)
