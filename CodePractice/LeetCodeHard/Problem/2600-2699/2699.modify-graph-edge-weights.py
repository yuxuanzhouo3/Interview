"""
LC 2699. Modify Graph Edge Weights — Hard
https://leetcode.com/problems/modify-graph-edge-weights/
Best: Dijkstra with deferred -1 assignment — O(n e log n)
"""

from typing import List
import heapq


class Solution:
    def modifiedGraphEdges(
        self, n: int, edges: List[List[int]], source: int, destination: int, target: int
    ) -> List[List[int]]:
        INF = 2 * 10**9
        g = [[] for _ in range(n)]
        for i, (u, v, _) in enumerate(edges):
            g[u].append((v, i))
            g[v].append((u, i))

        def dijkstra(op: int) -> List[int]:
            # op=0: ignore -1 (weight INF); op=1: treat -1 as 1
            dist = [10**18] * n
            dist[source] = 0
            pq = [(0, source)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, ei in g[u]:
                    w = edges[ei][2]
                    if w == -1:
                        w = 1 if op else INF
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            return dist

        d_ignore = dijkstra(0)
        if d_ignore[destination] < target:
            return []
        if d_ignore[destination] == target:
            for e in edges:
                if e[2] == -1:
                    e[2] = INF
            return edges

        # distances from destination treating -1 as 1
        dist_dest = [10**18] * n
        dist_dest[destination] = 0
        pq = [(0, destination)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist_dest[u]:
                continue
            for v, ei in g[u]:
                w = edges[ei][2]
                if w == -1:
                    w = 1
                nd = d + w
                if nd < dist_dest[v]:
                    dist_dest[v] = nd
                    heapq.heappush(pq, (nd, v))
        if dist_dest[source] > target:
            return []

        dist = [10**18] * n
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, ei in g[u]:
                w = edges[ei][2]
                if w == -1:
                    w = max(1, target - d - dist_dest[v])
                    edges[ei][2] = w
                else:
                    w = edges[ei][2]
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        for e in edges:
            if e[2] == -1:
                e[2] = INF

        # final check
        dist_f = dijkstra(1)  # no -1 left
        # rebuild dijkstra on final weights
        dist_f = [10**18] * n
        dist_f[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist_f[u]:
                continue
            for v, ei in g[u]:
                nd = d + edges[ei][2]
                if nd < dist_f[v]:
                    dist_f[v] = nd
                    heapq.heappush(pq, (nd, v))
        if dist_f[destination] != target:
            return []
        return edges


def _shortest(n, edges, s, t):
    g = [[] for _ in range(n)]
    for u, v, w in edges:
        g[u].append((v, w))
        g[v].append((u, w))
    dist = [10**18] * n
    dist[s] = 0
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in g[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist[t]


def _test():
    s = Solution()
    e1 = [[4, 1, -1], [2, 0, -1], [0, 3, -1], [4, 3, -1]]
    r1 = s.modifiedGraphEdges(5, [x[:] for x in e1], 0, 1, 5)
    assert r1 and _shortest(5, r1, 0, 1) == 5
    assert s.modifiedGraphEdges(3, [[0, 1, -1], [0, 2, 5]], 0, 2, 6) == []
    e3 = [[1, 0, 4], [1, 2, 3], [2, 3, 5], [0, 3, -1]]
    r3 = s.modifiedGraphEdges(4, [x[:] for x in e3], 0, 2, 6)
    assert r3 and _shortest(4, r3, 0, 2) == 6
    print("LC 2699 OK")


if __name__ == "__main__":
    _test()
