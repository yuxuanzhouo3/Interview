"""
LC 736. Parse Lisp Expression — Hard
Best: Recursive descent with scope stack — O(n)
"""

from typing import List, Dict


class Solution:
    def evaluate(self, expression: str) -> int:
        def parse(s, i):
            # returns (value or tokens handled via eval_expr)
            pass

        def tokenize(s):
            s = s.replace("(", " ( ").replace(")", " ) ")
            return s.split()

        tokens = tokenize(expression)
        self.i = 0

        def peek():
            return tokens[self.i] if self.i < len(tokens) else None

        def consume():
            t = tokens[self.i]
            self.i += 1
            return t

        def parse_expr(scopes: List[Dict[str, int]]) -> int:
            if peek() != "(":
                t = consume()
                if t.lstrip("-").isdigit():
                    return int(t)
                for sc in reversed(scopes):
                    if t in sc:
                        return sc[t]
                raise ValueError(t)
            consume()  # (
            op = consume()
            if op == "let":
                local = {}
                scopes.append(local)
                while peek() != ")" and not (peek() == ")" ):
                    # if next is last expr before )
                    # look ahead: if only one token left before ) it's expr
                    # better: while next is var and after that not )
                    if peek() == ")":
                        break
                    # check if remaining is single expr: if current is ( or digit/var and after parse only )
                    # standard approach: while there are pairs
                    # Peek structure: let v1 e1 v2 e2 ... expr )
                    # We don't know until we see if after var comes ) 
                    name = peek()
                    # if next token after name is ), then name is actually expr
                    # save pos
                    # Actually parse: repeatedly if look like var binding
                    if name == "(" or (name.lstrip("-").isdigit()):
                        val = parse_expr(scopes)
                        consume()  # )
                        scopes.pop()
                        return val
                    # name is variable
                    consume()
                    if peek() == ")":
                        # name was expression (variable ref)
                        for sc in reversed(scopes):
                            if name in sc:
                                val = sc[name]
                                break
                        else:
                            val = int(name) if name.lstrip("-").isdigit() else (_ for _ in ()).throw(ValueError(name))
                        consume()
                        scopes.pop()
                        return val
                    local[name] = parse_expr(scopes)
                # shouldn't reach
            elif op == "add":
                a = parse_expr(scopes)
                b = parse_expr(scopes)
                consume()  # )
                return a + b
            elif op == "mult":
                a = parse_expr(scopes)
                b = parse_expr(scopes)
                consume()  # )
                return a * b
            raise ValueError(op)

        # cleaner rewrite of let
        self.i = 0

        def eval_expr(env: Dict[str, int]) -> int:
            tok = consume()
            if tok != "(":
                if tok.lstrip("-").isdigit():
                    return int(tok)
                return env[tok]
            op = consume()
            if op == "let":
                new_env = env.copy()
                while True:
                    if peek() == "(":
                        # expression then )
                        val = eval_expr(new_env)
                        consume()
                        return val
                    var = consume()
                    if peek() == ")":
                        consume()
                        if var.lstrip("-").isdigit():
                            return int(var)
                        return new_env[var]
                    new_env[var] = eval_expr(new_env)
            elif op == "add":
                a = eval_expr(env)
                b = eval_expr(env)
                consume()
                return a + b
            elif op == "mult":
                a = eval_expr(env)
                b = eval_expr(env)
                consume()
                return a * b
            raise ValueError(op)

        return eval_expr({})


def _test():
    s = Solution()
    assert s.evaluate("(let x 2 (mult x (let x 3 y 4 (add x y))))") == 14
    assert s.evaluate("(let x 3 x 2 x)") == 2
    assert s.evaluate("(let x 1 y 2 x (add x y) (add x y))") == 5
    print("LC 736 OK")


if __name__ == "__main__":
    _test()
