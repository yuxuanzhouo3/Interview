"""
LC 3768. Minimum Inversion Count in Subarrays of Fixed Length — Hard
https://leetcode.com/problems/minimum-inversion-count-in-subarrays-of-fixed-length/
Best: Sliding window + Fenwick inversions — O(n log n)
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


class _Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

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


class Solution:
    def minInversionCount(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k <= 1:
            return 0
        vals = sorted(set(nums))
        rank = {v: i + 1 for i, v in enumerate(vals)}
        m = len(vals)
        fw = _Fenwick(m)
        inv = 0
        for i in range(k):
            r = rank[nums[i]]
            inv += i - fw.sum(r)
            fw.add(r, 1)
        best = inv
        for i in range(k, n):
            r_out = rank[nums[i - k]]
            inv -= fw.sum(r_out - 1)
            fw.add(r_out, -1)
            r_in = rank[nums[i]]
            inv += (k - 1) - fw.sum(r_in)
            fw.add(r_in, 1)
            best = min(best, inv)
        return best


def _test():

    s = Solution()
    assert s.minInversionCount([3, 1, 2, 5, 4], 3) == 0
    assert s.minInversionCount([5, 3, 2, 1], 4) == 6
    assert s.minInversionCount([2, 1], 1) == 0

    print("LC 3768 OK")


if __name__ == "__main__":
    _test()
