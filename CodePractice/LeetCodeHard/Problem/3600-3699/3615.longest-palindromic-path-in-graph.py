"""
LC 3615. Longest Palindromic Path in Graph — Hard
https://leetcode.com/problems/longest-palindromic-path-in-graph/
Best: Bitmask DP expand palindrome ends — O(2^n n^2)
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
    def maxLen(self, n: int, edges: List[List[int]], label: str) -> int:
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        # reachable[mask][i][j] too heavy; use set of (mask,i,j)
        # n small typically <= 14
        from collections import deque
        seen = set()
        q = deque()
        ans = 1
        for i in range(n):
            q.append((1 << i, i, i))
            seen.add((1 << i, i, i))
        for i in range(n):
            for j in g[i]:
                if i < j and label[i] == label[j]:
                    mask = (1 << i) | (1 << j)
                    q.append((mask, i, j))
                    seen.add((mask, i, j))
                    ans = max(ans, 2)
        while q:
            mask, i, j = q.popleft()
            ans = max(ans, bin(mask).count("1"))
            for ni in g[i]:
                if mask & (1 << ni):
                    continue
                for nj in g[j]:
                    if ni == nj or (mask & (1 << nj)):
                        continue
                    if label[ni] != label[nj]:
                        continue
                    nmask = mask | (1 << ni) | (1 << nj)
                    state = (nmask, ni, nj)
                    if state not in seen:
                        seen.add(state)
                        q.append(state)
        return ans


def _test():

    s = Solution()
    assert s.maxLen(3, [[0, 1], [1, 2]], "aba") == 3
    assert s.maxLen(3, [[0, 1], [0, 2]], "abc") == 1
    assert s.maxLen(4, [[0, 2], [0, 3], [3, 1]], "bbac") >= 1

    print("LC 3615 OK")


if __name__ == "__main__":
    _test()
