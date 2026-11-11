"""
AdvancedGraph / Shortest Path templates (Dijkstra + Topo DAG)
Template: query(dist) / update(relax) / add(edge) / del(edge rebuild)
App: 路由、地图导航、网络延时 LC 743。
"""

import heapq
from collections import defaultdict


class WeightedGraph:
    def __init__(self):
        self.adj = defaultdict(dict)

    def add(self, u, v, w):
        self.adj[u][v] = w

    def update(self, u, v, w):
        if v not in self.adj[u]:
            return False
        self.adj[u][v] = w
        return True

    def delete(self, u, v):
        self.adj[u].pop(v, None)

    def query(self, src, dst=None):
        dist = self.dijkstra(src)
        return dist if dst is None else dist.get(dst, float("inf"))

    def dijkstra(self, src):
        dist = {src: 0}
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, float("inf")):
                continue
            for v, w in self.adj[u].items():
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist


def _test():
    g = WeightedGraph()
    for e in [(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 6), (2, 3, 3)]:
        g.add(*e)
    assert g.query(0, 3) == 6
    assert g.update(1, 3, 1)
    assert g.query(0, 3) == 2
    g.delete(1, 3)
    assert g.query(0, 3) == 6
    print("ShortestPath OK", g.query(0))


APPLICATION = """
- 地图/游戏寻路、BGP 简化模型、微服务调用延时。
- 负权 → Bellman-Ford；全源 → Floyd；DAG → 拓扑+松弛。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
