"""
LC 770. Basic Calculator IV — Hard
https://leetcode.com/problems/basic-calculator-iv/
Best: Recursive descent → polynomial Counter — O(n · T)
"""

from typing import List
from collections import Counter


class Solution:
    def basicCalculatorIV(
        self, expression: str, evalvars: List[str], evalints: List[int]
    ) -> List[str]:
        known = dict(zip(evalvars, evalints))

        def combine(a: Counter, b: Counter, op: str) -> Counter:
            res = Counter(a)
            if op == "+":
                for k, v in b.items():
                    res[k] += v
            elif op == "-":
                for k, v in b.items():
                    res[k] -= v
            else:  # *
                res = Counter()
                for k1, v1 in a.items():
                    for k2, v2 in b.items():
                        key = tuple(sorted(k1 + k2))
                        res[key] += v1 * v2
            return Counter({k: v for k, v in res.items() if v})

        s = expression.replace(" ", "")
        i = 0

        def parse_expr():
            nonlocal i
            res = parse_term()
            while i < len(s) and s[i] in "+-":
                op = s[i]
                i += 1
                res = combine(res, parse_term(), op)
            return res

        def parse_term():
            nonlocal i
            res = parse_factor()
            while i < len(s) and s[i] == "*":
                i += 1
                res = combine(res, parse_factor(), "*")
            return res

        def parse_factor():
            nonlocal i
            if s[i] == "(":
                i += 1
                res = parse_expr()
                i += 1  # )
                return res
            if s[i] == "-":
                i += 1
                return combine(Counter(), parse_factor(), "-")
            j = i
            if s[i].isdigit():
                while j < len(s) and s[j].isdigit():
                    j += 1
                val = int(s[i:j])
                i = j
                return Counter({(): val})
            while j < len(s) and s[j].isalpha():
                j += 1
            name = s[i:j]
            i = j
            if name in known:
                return Counter({(): known[name]})
            return Counter({(name,): 1})

        poly = parse_expr()
        # sort: higher degree first, then lex of vars
        items = sorted(
            ((k, v) for k, v in poly.items() if v),
            key=lambda kv: (-len(kv[0]), kv[0]),
        )
        out = []
        for vars_, coef in items:
            if vars_:
                out.append(f"{coef}*{'*'.join(vars_)}")
            else:
                out.append(str(coef))
        return out


def _test():
    s = Solution()
    assert s.basicCalculatorIV("e + 8 - a + 5", ["e"], [1]) == ["-1*a", "14"]
    assert s.basicCalculatorIV("e - 8 + temperature - pressure", ["e", "temperature"], [1, 12]) == [
        "-1*pressure",
        "5",
    ]
    assert s.basicCalculatorIV("(e + 8) * (e - 8)", [], []) == ["1*e*e", "-64"]
    assert s.basicCalculatorIV("7 - 7", [], []) == []
    assert s.basicCalculatorIV("a * b * c + b * a * c * 4", [], []) == ["5*a*b*c"]
    print("LC 770 OK")


if __name__ == "__main__":
    _test()
