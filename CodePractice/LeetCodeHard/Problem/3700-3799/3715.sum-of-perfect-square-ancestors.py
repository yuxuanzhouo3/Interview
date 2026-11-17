"""
LC 3715. Sum of Perfect Square Ancestors — Hard
https://leetcode.com/problems/sum-of-perfect-square-ancestors/
Best: DFS + square-free kernel counting — O(n sqrt A)
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
    def sumOfAncestors(self, n: int, edges: List[List[int]], nums: List[int]) -> int:
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        def square_free_kernel(x: int) -> int:
            res = 1
            d = 2
            while d * d <= x:
                c = 0
                while x % d == 0:
                    x //= d
                    c += 1
                if c & 1:
                    res *= d
                d += 1
            if x > 1:
                res *= x
            return res

        ker = [square_free_kernel(v) for v in nums]
        ans = 0
        cnt: Dict[int, int] = defaultdict(int)

        def dfs(u: int, p: int) -> None:
            nonlocal ans
            ans += cnt[ker[u]]
            cnt[ker[u]] += 1
            for v in g[u]:
                if v != p:
                    dfs(v, u)
            cnt[ker[u]] -= 1

        dfs(0, -1)
        return ans


def _test():

    s = Solution()
    assert s.sumOfAncestors(3, [[0, 1], [1, 2]], [2, 8, 2]) == 3
    assert s.sumOfAncestors(3, [[0, 1], [0, 2]], [1, 2, 4]) == 1

    print("LC 3715 OK")


if __name__ == "__main__":
    _test()
