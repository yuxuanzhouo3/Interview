"""
LC 3624. Number of Integers With Popcount-Depth Equal to K II — Hard
https://leetcode.com/problems/number-of-integers-with-popcount-depth-equal-to-k-ii/
Best: Per-depth Fenwick for point update / range count — O(q log n)
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
        self.bit = [0] * (n + 1)

    def add(self, i: int, v: int) -> None:
        i += 1
        while i <= self.n:
            self.bit[i] += v
            i += i & -i

    def sum(self, i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def range_sum(self, l: int, r: int) -> int:
        return self.sum(r) - (self.sum(l - 1) if l else 0)


class Solution:
    def popcountDepth(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def depth(x: int) -> int:
            if x == 1:
                return 0
            d = 0
            while x != 1:
                x = bin(x).count("1")
                d += 1
            return d

        n = len(nums)
        trees = [_BIT(n) for _ in range(6)]
        cur = [0] * n
        for i, v in enumerate(nums):
            d = depth(v)
            cur[i] = d
            if d <= 5:
                trees[d].add(i, 1)
        ans = []
        for q in queries:
            if q[0] == 1:
                _, l, r, k = q
                ans.append(trees[k].range_sum(l, r) if k <= 5 else 0)
            else:
                _, idx, val = q
                old = cur[idx]
                if old <= 5:
                    trees[old].add(idx, -1)
                nd = depth(val)
                cur[idx] = nd
                if nd <= 5:
                    trees[nd].add(idx, 1)
        return ans


def _test():

    s = Solution()
    assert s.popcountDepth([2, 4], [[1, 0, 1, 1], [2, 1, 1], [1, 0, 1, 0]]) == [2, 1]
    assert s.popcountDepth([3, 5, 6], [[1, 0, 2, 2], [2, 1, 4], [1, 1, 2, 1], [1, 0, 1, 0]]) == [3, 1, 0]

    print("LC 3624 OK")


if __name__ == "__main__":
    _test()
