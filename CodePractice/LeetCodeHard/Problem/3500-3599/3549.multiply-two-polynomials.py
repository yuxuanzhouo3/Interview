"""
LC 3549. Multiply Two Polynomials — Hard
https://leetcode.com/problems/multiply-two-polynomials/
Best: FFT convolution — O(n log n)
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
    def multiply(self, poly1: List[int], poly2: List[int]) -> List[int]:
        if not poly1 or not poly2:
            return []
        m = len(poly1) + len(poly2) - 1
        n = 1
        while n < m:
            n <<= 1
        fa = list(map(complex, poly1)) + [0j] * (n - len(poly1))
        fb = list(map(complex, poly2)) + [0j] * (n - len(poly2))
        self._fft(fa, False)
        self._fft(fb, False)
        for i in range(n):
            fa[i] *= fb[i]
        self._fft(fa, True)
        return [int(round(fa[i].real)) for i in range(m)]

    def _fft(self, a: List[complex], invert: bool) -> None:
        n = len(a)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                a[i], a[j] = a[j], a[i]
        length = 2
        while length <= n:
            ang = 2 * math.pi / length * (-1 if invert else 1)
            wlen = complex(math.cos(ang), math.sin(ang))
            for i in range(0, n, length):
                w = 1 + 0j
                for j in range(i, i + length // 2):
                    u, v = a[j], a[j + length // 2] * w
                    a[j], a[j + length // 2] = u + v, u - v
                    w *= wlen
            length <<= 1
        if invert:
            for i in range(n):
                a[i] /= n


def _test():

    s = Solution()
    assert s.multiply([3, 2, 5], [1, 4]) == [3, 14, 13, 20]
    assert s.multiply([1, 0, -2], [-1]) == [-1, 0, 2]

    print("LC 3549 OK")


if __name__ == "__main__":
    _test()
