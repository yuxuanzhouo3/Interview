"""
LC 3735. Lexicographically Smallest String After Reverse II — Hard
https://leetcode.com/problems/lexicographically-smallest-string-after-reverse-ii/
Best: Enumerate all k reverse options — O(n^2)
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
    def lexSmallest(self, s: str) -> str:
        n = len(s)
        best = None
        for k in range(1, n + 1):
            a = s[:k][::-1] + s[k:]
            b = s[: n - k] + s[n - k :][::-1]
            cand = a if a < b else b
            if best is None or cand < best:
                best = cand
        return best


def _test():

    s = Solution()
    assert s.lexSmallest("dcab") == "acdb"
    assert s.lexSmallest("abba") == "aabb"
    assert s.lexSmallest("zxy") == "xzy"

    print("LC 3735 OK")


if __name__ == "__main__":
    _test()
