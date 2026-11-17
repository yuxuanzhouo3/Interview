"""
LC 3725. Count Ways to Choose Coprime Integers from Rows — Hard
https://leetcode.com/problems/count-ways-to-choose-coprime-integers-from-rows/
Best: DP on running GCD — O(m * n * V log V)
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
    def countCoprime(self, mat: List[List[int]]) -> int:
        MOD = 10**9 + 7
        dp = Counter({0: 1})
        for row in mat:
            ndp: Counter = Counter()
            for g, ways in dp.items():
                for x in row:
                    ng = x if g == 0 else gcd(g, x)
                    ndp[ng] = (ndp[ng] + ways) % MOD
            dp = ndp
        return dp[1] % MOD


def _test():

    s = Solution()
    assert s.countCoprime([[1, 2], [3, 4]]) == 3
    assert s.countCoprime([[2, 2], [2, 2]]) == 0

    print("LC 3725 OK")


if __name__ == "__main__":
    _test()
