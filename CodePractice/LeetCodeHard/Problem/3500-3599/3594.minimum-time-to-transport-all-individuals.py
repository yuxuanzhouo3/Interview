"""
LC 3594. Minimum Time to Transport All Individuals — Hard
https://leetcode.com/problems/minimum-time-to-transport-all-individuals/
Best: Dijkstra on (side, stage, mask) + submask enum — O(m 3^n log)
"""

from __future__ import annotations

import heapq
from typing import List


class Solution:
    def minTime(
        self, n: int, k: int, m: int, time: List[int], mul: List[float]
    ) -> float:
        popcount = [0] * (1 << n)
        for i in range(1, 1 << n):
            popcount[i] = popcount[i >> 1] + (i & 1)
        lookup = [
            max((time[i] for i in range(n) if mask & (1 << i)), default=0)
            for mask in range(1 << n)
        ]
        INF = float("inf")
        dist = [[[INF] * (1 << n) for _ in range(m)] for _ in range(2)]
        dist[0][0][(1 << n) - 1] = 0.0
        pq = [(0.0, 0, 0, (1 << n) - 1)]  # d, side, stage, mask_remaining_start

        def update(d: float, r: int, s: int, mask: int, submask: int) -> None:
            t = lookup[submask] * mul[s]
            nr = r ^ 1
            ns = (s + int(t)) % m
            new_mask = mask ^ submask
            nd = d + t
            if dist[nr][ns][new_mask] > nd:
                dist[nr][ns][new_mask] = nd
                heapq.heappush(pq, (nd, nr, ns, new_mask))

        while pq:
            d, r, s, mask = heapq.heappop(pq)
            if d != dist[r][s][mask]:
                continue
            if mask == 0:
                return d
            if r == 0:
                submask = mask
                while submask:
                    if popcount[submask] <= k:
                        update(d, r, s, mask, submask)
                    submask = (submask - 1) & mask
            else:
                for i in range(n):
                    if mask & (1 << i):
                        continue
                    update(d, r, s, mask, 1 << i)
        return -1.0


def _test():
    s = Solution()
    # n=1,k=1,m=2 — single person crosses once
    ans = s.minTime(1, 1, 2, [5], [1.0, 1.3])
    assert abs(ans - 5.0) < 1e-5
    # k=1,n=2 无解（船无法留在对岸）；k=2 可一次全过
    assert s.minTime(2, 1, 3, [3, 5], [1.0, 1.5, 0.8]) == -1.0
    ans2 = s.minTime(2, 2, 3, [3, 5], [1.0, 1.5, 0.8])
    assert abs(ans2 - 5.0) < 1e-5
    print("LC 3594 OK")


if __name__ == "__main__":
    _test()
