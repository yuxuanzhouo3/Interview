"""
LC 3710. Maximum Partition Factor — Hard
https://leetcode.com/problems/maximum-partition-factor/
Best: Binary search + bipartite check — O(n^2 log W)
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
    def maxPartitionFactor(self, points: List[List[int]]) -> int:
        n = len(points)
        if n <= 2:
            return 0
        dist = [[0] * n for _ in range(n)]
        mx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                dist[i][j] = dist[j][i] = d
                mx = max(mx, d)

        def ok(thr: int) -> bool:
            g = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if dist[i][j] < thr:
                        g[i].append(j)
                        g[j].append(i)
            color = [-1] * n

            def bfs(s: int) -> bool:
                q = [s]
                color[s] = 0
                for u in q:
                    for v in g[u]:
                        if color[v] == -1:
                            color[v] = color[u] ^ 1
                            q.append(v)
                        elif color[v] == color[u]:
                            return False
                return True

            for i in range(n):
                if color[i] == -1 and not bfs(i):
                    return False
            return True

        lo, hi = 0, mx + 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


def _test():

    s = Solution()
    assert s.maxPartitionFactor([[0, 0], [0, 2], [2, 0], [2, 2]]) == 4
    assert s.maxPartitionFactor([[0, 0], [0, 1], [10, 0]]) == 11

    print("LC 3710 OK")


if __name__ == "__main__":
    _test()
