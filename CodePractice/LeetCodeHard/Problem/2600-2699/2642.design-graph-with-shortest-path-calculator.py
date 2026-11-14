"""
LC 2642. Design Graph With Shortest Path Calculator — Hard
https://leetcode.com/problems/design-graph-with-shortest-path-calculator/
Best: Adjacency list + Dijkstra per query — O((n+e) log n)
"""

from typing import List
import heapq
from collections import defaultdict


class Graph:
    def __init__(self, n: int, edges: List[List[int]]):
        self.n = n
        self.g = defaultdict(list)
        for u, v, w in edges:
            self.g[u].append((v, w))

    def addEdge(self, edge: List[int]) -> None:
        u, v, w = edge
        self.g[u].append((v, w))

    def shortestPath(self, node1: int, node2: int) -> int:
        dist = [10**18] * self.n
        dist[node1] = 0
        pq = [(0, node1)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == node2:
                return d
            for v, w in self.g[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return -1


def _test():
    g = Graph(4, [[0, 2, 5], [0, 1, 2], [1, 2, 1], [3, 0, 3]])
    assert g.shortestPath(3, 2) == 6
    assert g.shortestPath(0, 3) == -1
    g.addEdge([1, 3, 4])
    assert g.shortestPath(0, 3) == 6
    print("LC 2642 OK")


if __name__ == "__main__":
    _test()
