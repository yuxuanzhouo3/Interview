"""
LC 3749. Evaluate Valid Expressions — Hard
https://leetcode.com/problems/evaluate-valid-expressions/
Best: Recursive descent — O(n)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple
import math
import heapq
import bisect
from collections import defaultdict, Counter, deque
from functools import lru_cache, reduce
from math import gcd, isqrt
from itertools import combinations, permutations, product


class Solution:
    def evaluateExpression(self, expression: str) -> int:
        def parse(i: int):
            if expression[i].isdigit() or expression[i] == "-":
                j = i + (expression[i] == "-")
                while j < len(expression) and expression[j].isdigit():
                    j += 1
                return int(expression[i:j]), j
            j = i
            while expression[j] != "(":
                j += 1
            op = expression[i:j]
            j += 1
            a, j = parse(j)
            j += 1
            b, j = parse(j)
            j += 1
            if op == "add":
                res = a + b
            elif op == "sub":
                res = a - b
            elif op == "mul":
                res = a * b
            else:
                res = a // b
            return res, j

        return parse(0)[0]


def _test():

    s = Solution()
    assert s.evaluateExpression("add(2,3)") == 5
    assert s.evaluateExpression("-42") == -42
    assert s.evaluateExpression("div(mul(4,sub(9,5)),add(1,1))") == 8

    print("LC 3749 OK")


if __name__ == "__main__":
    _test()
