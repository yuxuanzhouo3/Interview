"""
AdvancedGraph / MST（Kruskal + Prim）
Template: query(total|edges) / update / add / del
App: 光纤铺设、聚类、LC 1135 / 1584；Kruskal 用 DSU。
来源: CodePractice/graph.py §5
"""

import heapq
from collections import defaultdict


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


class MST:
    def __init__(self, n=0):
        self.n = n
        self.edges = []  # (u,v,w)
        self._key = {}

    def add(self, u, v=None, w=0):
        if v is None:
            self.n = max(self.n, u + 1)
            return
        self.n = max(self.n, u + 1, v + 1)
        key = (min(u, v), max(u, v))
        if key in self._key:
            self.edges[self._key[key]] = (u, v, w)
            return
        self._key[key] = len(self.edges)
        self.edges.append((u, v, w))

    def update(self, u, v, w):
        key = (min(u, v), max(u, v))
        if key not in self._key:
            return False
        self.edges[self._key[key]] = (u, v, w)
        return True

    def delete(self, u, v):
        key = (min(u, v), max(u, v))
        if key not in self._key:
            return
        i = self._key.pop(key)
        self.edges[i] = None
        self.edges = [e for e in self.edges if e is not None]
        self._key = {(min(a, b), max(a, b)): k for k, (a, b, _) in enumerate(self.edges)}

    def query(self, algo="kruskal"):
        if algo == "kruskal":
            return self.kruskal()
        if algo == "prim":
            return self.prim()
        raise ValueError(algo)

    def kruskal(self):
        edges = sorted(self.edges, key=lambda x: x[2])
        dsu = DSU(self.n)
        total, used, mst = 0, 0, []
        for u, v, w in edges:
            if dsu.union(u, v):
                total += w
                used += 1
                mst.append((u, v, w))
                if used == self.n - 1:
                    break
        if used != self.n - 1:
            return None
        return total, mst

    def prim(self, start=0):
        adj = defaultdict(list)
        for u, v, w in self.edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        visited = [False] * self.n
        pq = [(0, start, -1)]
        total, used, mst = 0, 0, []
        while pq:
            w, u, parent = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            total += w
            used += 1
            if parent != -1:
                mst.append((parent, u, w))
            for v, cost in adj[u]:
                if not visited[v]:
                    heapq.heappush(pq, (cost, v, u))
        if used != self.n:
            return None
        return total, mst


def _test():
    g = MST(4)
    for e in [(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 6), (2, 3, 3)]:
        g.add(*e)
    t1, _ = g.query("kruskal")
    t2, _ = g.query("prim")
    assert t1 == t2 == 6
    assert g.update(2, 3, 10)
    assert g.query("kruskal")[0] == 9  # 0-1-2 + 1-3 = 1+2+6
    g.delete(1, 3)
    # 仍连通: 0-1,1-2,2-3=10 → 13
    assert g.query("kruskal")[0] == 13
    print("MST OK", t1)


APPLICATION = """
- Kruskal: 排边 + DSU，稀疏图；Prim: 堆扩树，稠密也行。
- LC 1135 Connecting Cities；1584 Min Cost Connect Points。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
