"""
LC 3782. Last Remaining Integer After Alternating Deletion Operations — Hard
https://leetcode.com/problems/last-remaining-integer-after-alternating-deletion-operations/
Best: Closed-form alternating elimination — O(log n)
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
    def lastInteger(self, n: int) -> int:
        left = True
        start, step, cnt = 1, 1, n
        while cnt > 1:
            if (not left) and cnt % 2 == 0:
                start += step
            step *= 2
            cnt = (cnt + 1) // 2
            left = not left
        return start


def _test():

    s = Solution()
    assert s.lastInteger(8) == 3
    assert s.lastInteger(5) == 1
    assert s.lastInteger(1) == 1

    print("LC 3782 OK")


if __name__ == "__main__":
    _test()
