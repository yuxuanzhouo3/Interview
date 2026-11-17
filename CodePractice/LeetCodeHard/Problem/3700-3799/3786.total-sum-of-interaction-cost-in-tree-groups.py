"""
LC 3786. Total Sum of Interaction Cost in Tree Groups — Hard
https://leetcode.com/problems/total-sum-of-interaction-cost-in-tree-groups/
Best: Per-group tree DP distance sum — O(n * G)
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
    def interactionCosts(self, n: int, edges: List[List[int]], group: List[int]) -> int:
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        # sum of distances for pairs in same group
        # for each group, count nodes; use reroot or: for each edge, contrib = cnt_left * cnt_right for that group
        groups = sorted(set(group))
        ans = 0
        for gr in groups:
            is_g = [group[i] == gr for i in range(n)]
            total = sum(is_g)
            if total <= 1:
                continue

            def dfs(u: int, p: int) -> int:
                nonlocal ans
                cnt = 1 if is_g[u] else 0
                for v in g[u]:
                    if v == p:
                        continue
                    sub = dfs(v, u)
                    # edge u-v contributes sub * (total-sub)
                    ans += sub * (total - sub)
                    cnt += sub
                return cnt

            dfs(0, -1)
        return ans


def _test():

    s = Solution()
    assert s.interactionCosts(3, [[0, 1], [1, 2]], [1, 1, 1]) == 4
    assert s.interactionCosts(3, [[0, 1], [1, 2]], [3, 2, 3]) == 2
    assert s.interactionCosts(2, [[0, 1]], [9, 8]) == 0

    print("LC 3786 OK")


if __name__ == "__main__":
    _test()
