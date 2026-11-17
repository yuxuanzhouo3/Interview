"""
LC 3123. Find Edges in Shortest Paths — Hard
https://leetcode.com/problems/find-edges-in-shortest-paths/
Best: Dijkstra from 0 and n-1 — O((n+m) log n)
"""

from typing import List
import heapq
from collections import defaultdict


class Solution:
    def findAnswer(self, n: int, edges: List[List[int]]) -> List[bool]:
        g = defaultdict(list)
        for i, (u, v, w) in enumerate(edges):
            g[u].append((v, w, i))
            g[v].append((u, w, i))

        def dijkstra(src: int) -> List[int]:
            dist = [float("inf")] * n
            dist[src] = 0
            pq = [(0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w, _ in g[u]:
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            return dist

        d0 = dijkstra(0)
        dn = dijkstra(n - 1)
        if d0[n - 1] == float("inf"):
            return [False] * len(edges)
        target = d0[n - 1]
        ans = [False] * len(edges)
        for i, (u, v, w) in enumerate(edges):
            if d0[u] + w + dn[v] == target or d0[v] + w + dn[u] == target:
                ans[i] = True
        return ans


def _test():
    s = Solution()
    assert s.findAnswer(6, [[0, 1, 4], [0, 2, 1], [1, 3, 2], [1, 4, 3], [1, 5, 1], [2, 3, 1], [3, 5, 3], [4, 5, 2]]) == [
        True, True, True, False, True, True, True, False
    ]
    assert s.findAnswer(4, [[2, 0, 1], [0, 1, 1], [0, 3, 4], [3, 2, 2]]) == [True, False, False, True]
    print("LC 3123 OK")


if __name__ == "__main__":
    _test()
