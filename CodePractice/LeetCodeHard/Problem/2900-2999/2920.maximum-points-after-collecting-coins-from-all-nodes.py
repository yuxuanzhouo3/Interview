"""
LC 2920. Maximum Points After Collecting Coins From All Nodes — Hard
https://leetcode.com/problems/maximum-points-after-collecting-coins-from-all-nodes/
Best: Tree DP with floor/2 count (≤14) — O(n * log C)
"""

from typing import List
from functools import lru_cache
from collections import defaultdict


class Solution:
    def maximumPoints(self, edges: List[List[int]], coins: List[int], k: int) -> int:
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        @lru_cache(None)
        def dfs(u: int, p: int, t: int) -> int:
            # t = times floored /2 so far (cap ~14)
            c = coins[u] >> t
            # option1: take c-k
            op1 = c - k
            # option2: take floor(c/2) and increase t
            op2 = c // 2
            for v in g[u]:
                if v == p:
                    continue
                op1 += dfs(v, u, t)
                op2 += dfs(v, u, min(t + 1, 14))
            return max(op1, op2)

        return dfs(0, -1, 0)


def _test():
    s = Solution()
    assert s.maximumPoints([[0, 1], [1, 2], [2, 3]], [10, 10, 3, 3], 5) == 11
    assert s.maximumPoints([[0, 1], [0, 2]], [8, 4, 4], 0) == 16
    print("LC 2920 OK")


if __name__ == "__main__":
    _test()
