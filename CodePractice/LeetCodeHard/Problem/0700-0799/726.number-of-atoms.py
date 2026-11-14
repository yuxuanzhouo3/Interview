"""
LC 726. Number of Atoms — Hard
Best: Stack of counters parse formula — O(n^2) worst
"""

from collections import Counter
import re


class Solution:
    def countOfAtoms(self, formula: str) -> str:
        n = len(formula)
        stack = [Counter()]
        i = 0
        while i < n:
            if formula[i] == "(":
                stack.append(Counter())
                i += 1
            elif formula[i] == ")":
                i += 1
                j = i
                while j < n and formula[j].isdigit():
                    j += 1
                mult = int(formula[i:j] or 1)
                top = stack.pop()
                for k, v in top.items():
                    stack[-1][k] += v * mult
                i = j
            else:
                j = i + 1
                while j < n and formula[j].islower():
                    j += 1
                name = formula[i:j]
                i = j
                while j < n and formula[j].isdigit():
                    j += 1
                cnt = int(formula[i:j] or 1)
                stack[-1][name] += cnt
                i = j
        parts = []
        for name in sorted(stack[-1]):
            c = stack[-1][name]
            parts.append(name + (str(c) if c > 1 else ""))
        return "".join(parts)


def _test():
    s = Solution()
    assert s.countOfAtoms("H2O") == "H2O"
    assert s.countOfAtoms("Mg(OH)2") == "H2MgO2"
    assert s.countOfAtoms("K4(ON(SO3)2)2") == "K4N2O14S4"
    print("LC 726 OK")


if __name__ == "__main__":
    _test()
