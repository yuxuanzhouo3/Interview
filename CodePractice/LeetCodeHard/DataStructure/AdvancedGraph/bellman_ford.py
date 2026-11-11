"""
AdvancedGraph / Bellman-Ford
Template: query(dist|neg_cycle) / update / add / del
App: 负权最短路、差分约束、汇率套利负环；LC 787 变体。
来源: CodePractice/graph.py §4.4
"""

from collections import defaultdict


class BellmanFord:
    def __init__(self):
        self.edges = []  # (u,v,w)
        self._idx = {}   # (u,v) -> index
        self.nodes = set()

    def add(self, u, v, w=0):
        self.nodes.add(u)
        self.nodes.add(v)
        if (u, v) in self._idx:
            self.edges[self._idx[(u, v)]] = (u, v, w)
            return
        self._idx[(u, v)] = len(self.edges)
        self.edges.append((u, v, w))

    def update(self, u, v, w):
        if (u, v) not in self._idx:
            return False
        self.edges[self._idx[(u, v)]] = (u, v, w)
        return True

    def delete(self, u, v):
        if (u, v) not in self._idx:
            return
        i = self._idx.pop((u, v))
        self.edges[i] = None
        # compact lazily on query
        self.edges = [e for e in self.edges if e is not None]
        self._idx = {(a, b): k for k, (a, b, _) in enumerate(self.edges)}

    def query(self, src, dst=None):
        """返回 dist；可达负环返回 None。dst 给定则返回标量或 None。"""
        dist = self.run(src)
        if dist is None:
            return None
        if dst is None:
            return dist
        return dist.get(dst, float("inf"))

    def run(self, start):
        INF = float("inf")
        dist = {n: INF for n in self.nodes}
        dist[start] = 0
        n = len(self.nodes)
        for _ in range(n - 1):
            changed = False
            for u, v, w in self.edges:
                if dist[u] == INF:
                    continue
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    changed = True
            if not changed:
                break
        for u, v, w in self.edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                return None
        return dist


def _test():
    g = BellmanFord()
    g.add(0, 1, 5)
    g.add(1, 2, -2)
    g.add(0, 2, 4)
    assert g.query(0, 2) == 3
    assert g.update(1, 2, -10)
    assert g.query(0, 2) == -5
    g.add(2, 1, 1)  # 负环 1↔2
    assert g.query(0) is None
    g.delete(2, 1)
    assert g.query(0, 2) == -5
    print("BellmanFord OK", g.query(0))


APPLICATION = """
- 负权 / 检负环；差分约束系统转最短路。
- SPFA 是队列优化版，最坏仍差；竞赛慎用。
- |V| 大且无负权 → Dijkstra。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
