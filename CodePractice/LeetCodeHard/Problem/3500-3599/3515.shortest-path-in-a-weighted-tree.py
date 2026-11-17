"""
LC 3515. Shortest Path in a Weighted Tree — Hard
https://leetcode.com/problems/shortest-path-in-a-weighted-tree/
Best: Euler tour + Fenwick for path weights — O((n+q) log n)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Set
import math
import heapq
import bisect
from collections import defaultdict, Counter, deque
from functools import lru_cache
from math import gcd, isqrt


class _BIT:
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 2)

    def add(self, i: int, v: int) -> None:
        while i <= self.n:
            self.bit[i] += v
            i += i & -i

    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def range_add(self, l: int, r: int, v: int) -> None:
        self.add(l, v)
        self.add(r + 1, -v)


class Solution:
    def treeQueries(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        g = [[] for _ in range(n + 1)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        parent = [0] * (n + 1)
        pw = [0] * (n + 1)  # edge weight parent->u
        tin = [0] * (n + 1)
        tout = [0] * (n + 1)
        timer = 0

        def dfs(u: int, p: int) -> None:
            nonlocal timer
            timer += 1
            tin[u] = timer
            for v, w in g[u]:
                if v == p:
                    continue
                parent[v] = u
                pw[v] = w
                dfs(v, u)
            tout[u] = timer

        dfs(1, 0)
        bit = _BIT(n)
        for u in range(2, n + 1):
            bit.range_add(tin[u], tout[u], pw[u])

        # map undirected edge to child
        def child_of(u: int, v: int) -> int:
            return v if parent[v] == u else u

        ans = []
        for q in queries:
            if q[0] == 1:
                _, u, v, w = q
                c = child_of(u, v)
                delta = w - pw[c]
                pw[c] = w
                bit.range_add(tin[c], tout[c], delta)
            else:
                x = q[1]
                ans.append(bit.sum(tin[x]))
        return ans


def _test():

    s = Solution()
    assert s.treeQueries(2, [[1, 2, 7]], [[2, 2], [1, 1, 2, 4], [2, 2]]) == [7, 4]
    assert s.treeQueries(3, [[1, 2, 2], [1, 3, 4]], [[2, 1], [2, 3], [1, 1, 3, 7], [2, 2], [2, 3]]) == [0, 4, 2, 7]

    print("LC 3515 OK")


if __name__ == "__main__":
    _test()
