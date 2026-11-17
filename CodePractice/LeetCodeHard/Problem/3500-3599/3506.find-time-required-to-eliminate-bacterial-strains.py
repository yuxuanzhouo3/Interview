"""
LC 3506. Find Time Required to Eliminate Bacterial Strains — Hard
https://leetcode.com/problems/find-time-required-to-eliminate-bacterial-strains/
Best: Huffman / min-heap reverse merge — O(n log n)
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
    def minEliminationTime(self, timeReq: List[int], splitTime: int) -> int:
        h = list(timeReq)
        heapq.heapify(h)
        while len(h) > 1:
            heapq.heappop(h)
            heapq.heappush(h, heapq.heappop(h) + splitTime)
        return h[0]


def _test():

    s = Solution()
    assert s.minEliminationTime([10, 4, 5], 2) == 12
    assert s.minEliminationTime([10, 4], 5) == 15

    print("LC 3506 OK")


if __name__ == "__main__":
    _test()
