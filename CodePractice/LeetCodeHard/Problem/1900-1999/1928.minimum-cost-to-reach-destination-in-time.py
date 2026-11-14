"""
LC 1928. Minimum Cost to Reach Destination in Time — Hard
https://leetcode.com/problems/minimum-cost-to-reach-destination-in-time/
Best: Dijkstra on (node, time) by cost — O(n T log)
"""

from typing import List
import heapq


class Solution:
    def minCost(self, maxTime: int, edges: List[List[int]], passingFee: List[int]) -> int:
        n = len(passingFee)
        g = [[] for _ in range(n)]
        for u, v, t in edges:
            g[u].append((v, t))
            g[v].append((u, t))
        INF = 10**18
        best = [[INF] * (maxTime + 1) for _ in range(n)]
        best[0][0] = passingFee[0]
        pq = [(passingFee[0], 0, 0)]  # cost, time, node
        while pq:
            cost, t, u = heapq.heappop(pq)
            if cost > best[u][t]:
                continue
            if u == n - 1:
                return cost
            for v, dt in g[u]:
                nt = t + dt
                if nt > maxTime:
                    continue
                nc = cost + passingFee[v]
                if nc < best[v][nt]:
                    best[v][nt] = nc
                    heapq.heappush(pq, (nc, nt, v))
        return -1


def _test():
    s = Solution()
    assert (
        s.minCost(
            30,
            [[0, 1, 10], [1, 2, 10], [2, 5, 10], [0, 3, 1], [3, 4, 10], [4, 5, 15]],
            [5, 1, 2, 20, 20, 3],
        )
        == 11
    )
    assert (
        s.minCost(
            29,
            [[0, 1, 10], [1, 2, 10], [2, 5, 10], [0, 3, 1], [3, 4, 10], [4, 5, 15]],
            [5, 1, 2, 20, 20, 3],
        )
        == 48
    )
    assert (
        s.minCost(
            25,
            [[0, 1, 10], [1, 2, 10], [2, 5, 10], [0, 3, 1], [3, 4, 10], [4, 5, 15]],
            [5, 1, 2, 20, 20, 3],
        )
        == -1
    )
    print("LC 1928 OK")


if __name__ == "__main__":
    _test()
