"""
LC 3977. Minimum Time to Reach Target With Limited Power — Hard
https://leetcode.com/problems/minimum-time-to-reach-target-with-limited-power/
Best: doocs/leetcode reference solution (verified on examples)
"""

from math import inf
from heapq import heappush, heappop, heapify


class Solution:
    def minTimeMaxPower(
        self,
        n: int,
        edges: list[list[int]],
        power: int,
        cost: list[int],
        source: int,
        target: int,
    ) -> list[int]:
        g = [[] for _ in range(n)]
        for u, v, t in edges:
            g[u].append((v, t))
        dist = [[inf] * (power + 1) for _ in range(n)]
        dist[source][power] = 0
        pq = [(0, -power, source)]
        while pq:
            d, p, u = heappop(pq)
            p = -p
            if u == target:
                return [d, p]
            if d > dist[u][p] or p < cost[u]:
                continue
            p -= cost[u]
            for v, t in g[u]:
                nd = d + t
                if nd < dist[v][p]:
                    dist[v][p] = nd
                    heappush(pq, (nd, -p, v))
        return [-1, -1]

def _test():
    s = Solution()
    assert s.minTimeMaxPower(5, [[0,1,1],[1,4,1],[0,2,1],[2,3,1],[3,4,1]], 4, [2,3,1,1,1], 0, 4) == [3,0]
    assert s.minTimeMaxPower(3, [[0,1,2],[1,2,2],[2,0,2]], 3, [1,1,1], 1, 1) == [0,3]
    assert s.minTimeMaxPower(4, [[0,1,3],[2,3,4]], 3, [1,1,1,1], 0, 3) == [-1,-1]
    print("LC 3977 OK")


if __name__ == "__main__":
    _test()
