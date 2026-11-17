"""
LC 882. Reachable Nodes In Subdivided Graph — Hard
https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/
Best: Dijkstra + edge remaining counts — O(E log V)
"""

from typing import List
import heapq
from collections import defaultdict


class Solution:
    def reachableNodes(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        g = defaultdict(dict)
        for u, v, cnt in edges:
            g[u][v] = g[v][u] = cnt

        dist = {0: 0}
        pq = [(0, 0)]  # used moves, node
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            for v, cnt in g[u].items():
                nd = d + cnt + 1
                if nd < dist.get(v, float("inf")) and nd <= maxMoves:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        ans = len(dist)
        for u, v, cnt in edges:
            a = max(0, maxMoves - dist[u]) if u in dist else 0
            b = max(0, maxMoves - dist[v]) if v in dist else 0
            ans += min(cnt, a + b)
        return ans


def _test():
    s = Solution()
    assert s.reachableNodes([[0, 1, 10], [0, 2, 1], [1, 2, 2]], 6, 3) == 13
    assert s.reachableNodes([[0, 1, 4], [1, 2, 6], [0, 2, 8], [1, 3, 1]], 10, 4) == 23
    assert s.reachableNodes([[1, 2, 4], [1, 4, 5], [1, 3, 1], [2, 3, 4], [3, 4, 5]], 17, 5) == 1
    print("LC 882 OK")


if __name__ == "__main__":
    _test()
