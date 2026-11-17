"""
LC 2714. Find Shortest Path with K Hops — Hard
Best: Dijkstra state (node, hops_used) — O(E k log)
"""

from typing import List
import heapq
from collections import defaultdict


class Solution:
    def shortestPathWithHops(self, n: int, edges: List[List[int]], s: int, d: int, k: int) -> int:
        g = defaultdict(list)
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        INF = 10**18
        dist = [[INF] * (k + 1) for _ in range(n)]
        dist[s][0] = 0
        pq = [(0, s, 0)]
        while pq:
            cost, u, used = heapq.heappop(pq)
            if cost > dist[u][used]:
                continue
            if u == d:
                return cost
            for v, w in g[u]:
                if cost + w < dist[v][used]:
                    dist[v][used] = cost + w
                    heapq.heappush(pq, (cost + w, v, used))
                if used < k and cost < dist[v][used + 1]:
                    dist[v][used + 1] = cost
                    heapq.heappush(pq, (cost, v, used + 1))
        return min(dist[d])


def _test():
    s = Solution()
    assert s.shortestPathWithHops(4, [[0, 1, 4], [0, 2, 2], [2, 3, 6]], 0, 3, 1) == 2
    print("LC 2714 OK")


if __name__ == "__main__":
    _test()
