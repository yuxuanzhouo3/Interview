"""
LC 2203. Minimum Weighted Subgraph With the Required Paths — Hard
https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/
Best: 3x Dijkstra meet at intermediate — O(E log V)
"""


from typing import List
from collections import defaultdict
import heapq


class Solution:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
        g = defaultdict(list)
        rg = defaultdict(list)
        for u, v, w in edges:
            g[u].append((v, w))
            rg[v].append((u, w))

        def dijkstra(graph, start):
            dist = [10**30] * n
            dist[start] = 0
            h = [(0, start)]
            while h:
                d, u = heapq.heappop(h)
                if d != dist[u]:
                    continue
                for v, w in graph[u]:
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(h, (nd, v))
            return dist

        d1 = dijkstra(g, src1)
        d2 = dijkstra(g, src2)
        dd = dijkstra(rg, dest)
        ans = 10**30
        for i in range(n):
            ans = min(ans, d1[i] + d2[i] + dd[i])
        return -1 if ans >= 10**30 else ans


def _test():
    s = Solution()
    assert s.minimumWeight(6, [[0, 2, 2], [0, 5, 6], [1, 0, 3], [1, 4, 5], [2, 1, 1], [2, 3, 3], [2, 3, 4], [3, 4, 2], [4, 5, 1]], 0, 1, 5) == 9
    assert s.minimumWeight(3, [[0, 1, 1], [2, 1, 1]], 0, 1, 2) == -1
    print("LC 2203 OK")


if __name__ == "__main__":
    _test()

