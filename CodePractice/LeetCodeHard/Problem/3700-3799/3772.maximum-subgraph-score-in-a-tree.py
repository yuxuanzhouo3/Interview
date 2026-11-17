"""
LC 3772. Maximum Subgraph Score in a Tree — Hard
https://leetcode.com/problems/maximum-subgraph-score-in-a-tree/
Best: Rerooting DP keep positive contributions — O(n)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Set
import math
import heapq
import bisect
from collections import defaultdict, Counter, deque
from functools import lru_cache
from math import gcd, isqrt


class Solution:
    def maxSubgraphScore(self, n: int, edges: List[List[int]], good: List[int]) -> List[int]:
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        val = [1 if good[i] else -1 for i in range(n)]
        down = [0] * n

        def dfs1(u: int, p: int) -> None:
            down[u] = val[u]
            for v in g[u]:
                if v == p:
                    continue
                dfs1(v, u)
                if down[v] > 0:
                    down[u] += down[v]

        dfs1(0, -1)
        ans = [0] * n

        def dfs2(u: int, p: int, up: int) -> None:
            # best at u: val[u] + sum positive child downs + positive up
            best = val[u] + (up if up > 0 else 0)
            for v in g[u]:
                if v == p:
                    continue
                if down[v] > 0:
                    best += down[v]
            ans[u] = best
            for v in g[u]:
                if v == p:
                    continue
                # contribution from u toward v
                excl = best - (down[v] if down[v] > 0 else 0)
                dfs2(v, u, excl)

        dfs2(0, -1, 0)
        return ans


def _test():

    s = Solution()
    assert s.maxSubgraphScore(3, [[0, 1], [1, 2]], [1, 0, 1]) == [1, 1, 1]
    assert s.maxSubgraphScore(2, [[0, 1]], [0, 0]) == [-1, -1]

    print("LC 3772 OK")


if __name__ == "__main__":
    _test()
