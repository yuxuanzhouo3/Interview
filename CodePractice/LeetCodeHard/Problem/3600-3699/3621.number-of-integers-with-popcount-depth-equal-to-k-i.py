"""
LC 3621. Number of Integers With Popcount-Depth Equal to K I — Hard
https://leetcode.com/problems/number-of-integers-with-popcount-depth-equal-to-k-i/
Best: Digit DP on popcount — O(log n * log log n)
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
    def popcountDepth(self, n: int, k: int) -> int:
        if k == 0:
            return int(n >= 1)

        def depth(x: int) -> int:
            d = 0
            while x != 1:
                x = bin(x).count("1")
                d += 1
            return d

        maxb = n.bit_length() + 2
        dep = [0] * (maxb + 1)
        for c in range(1, maxb + 1):
            dep[c] = depth(c)

        bits = bin(n)[2:]
        L = len(bits)

        @lru_cache(None)
        def dp(pos: int, ones: int, tight: bool, started: bool) -> int:
            if pos == L:
                if not started:
                    return 0
                return 1 if dep[ones] == k - 1 else 0
            up = int(bits[pos]) if tight else 1
            res = 0
            for b in range(up + 1):
                nt = tight and b == up
                ns = started or b == 1
                no = ones + b if ns else 0
                res += dp(pos + 1, no, nt, ns)
            return res

        ans = dp(0, 0, True, False)
        if k == 1 and n >= 1:
            ans -= 1  # number 1 has depth 0
        return ans


def _test():

    s = Solution()
    assert s.popcountDepth(4, 1) == 2
    assert s.popcountDepth(7, 2) == 3

    print("LC 3621 OK")


if __name__ == "__main__":
    _test()
