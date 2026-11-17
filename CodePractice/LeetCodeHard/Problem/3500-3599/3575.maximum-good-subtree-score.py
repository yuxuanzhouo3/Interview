"""
LC 3575. Maximum Good Subtree Score — Hard
https://leetcode.com/problems/maximum-good-subtree-score/
Best: Tree knapsack on digit masks — O(n * 2^10 * 10)
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
    def goodSubtreeSum(self, vals: List[int], par: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(vals)
        g = [[] for _ in range(n)]
        for i in range(1, n):
            g[par[i]].append(i)

        def mask_of(x: int) -> int:
            m = 0
            if x == 0:
                return 1  # digit 0
            while x:
                d = x % 10
                if m & (1 << d):
                    return -1
                m |= 1 << d
                x //= 10
            return m

        masks = [mask_of(v) for v in vals]
        ans = 0

        def dfs(u: int) -> List[int]:
            # dp[mask] = max score for subset with digit mask
            nonlocal ans
            dp = [-10**18] * (1 << 10)
            dp[0] = 0
            if masks[u] != -1:
                dp[masks[u]] = vals[u]
            for v in g[u]:
                child = dfs(v)
                ndp = dp[:]
                for m1 in range(1 << 10):
                    if dp[m1] <= -10**17:
                        continue
                    for m2 in range(1 << 10):
                        if child[m2] <= -10**17:
                            continue
                        if m1 & m2:
                            continue
                        ndp[m1 | m2] = max(ndp[m1 | m2], dp[m1] + child[m2])
                dp = ndp
            ans = (ans + max(dp)) % MOD
            return dp

        dfs(0)
        return ans % MOD


def _test():

    s = Solution()
    assert s.goodSubtreeSum([2, 3], [-1, 0]) == 8
    assert s.goodSubtreeSum([1, 5, 2], [-1, 0, 0]) == 15

    print("LC 3575 OK")


if __name__ == "__main__":
    _test()
