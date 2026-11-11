"""
ClassifyThinking / 离线查询
Template: query(answers) / update / add(edge|query) / del
App: 查询按限制排序，边按权扫入 DSU；LC 1697。
"""

from collections import defaultdict


class OfflineConnectivity:
    def __init__(self, n=0):
        self.n = n
        self.edges = []   # (w, u, v)
        self.queries = [] # (limit, u, v, idx)
        self.parent = list(range(n))

    def add(self, kind, *args):
        """add('edge', u, v, w) 或 add('query', u, v, limit)。"""
        if kind == "edge":
            u, v, w = args
            self.n = max(self.n, u + 1, v + 1)
            self.edges.append((w, u, v))
        elif kind == "query":
            u, v, limit = args
            self.n = max(self.n, u + 1, v + 1)
            self.queries.append((limit, u, v, len(self.queries)))
        else:
            raise ValueError(kind)
        self.parent = list(range(self.n))

    def update(self, kind, idx, *args):
        if kind == "edge" and 0 <= idx < len(self.edges):
            self.edges[idx] = (args[2], args[0], args[1])
            return True
        return False

    def delete(self, kind, idx):
        if kind == "edge" and 0 <= idx < len(self.edges):
            self.edges.pop(idx)
        elif kind == "query" and 0 <= idx < len(self.queries):
            self.queries.pop(idx)
            self.queries = [(l, u, v, i) for i, (l, u, v, _) in enumerate(self.queries)]

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def query(self):
        self.parent = list(range(self.n))
        edges = sorted(self.edges)
        qs = sorted(self.queries)
        ans = [False] * len(self.queries)
        j = 0
        for limit, u, v, idx in qs:
            while j < len(edges) and edges[j][0] <= limit:
                _, a, b = edges[j]
                self.union(a, b)
                j += 1
            ans[idx] = self.find(u) == self.find(v)
        return ans


def _test():
    g = OfflineConnectivity()
    g.add("edge", 0, 1, 2)
    g.add("edge", 1, 2, 4)
    g.add("edge", 0, 2, 8)
    g.add("query", 0, 2, 4)  # via 0-1-2 weights 2,4 → yes
    g.add("query", 0, 2, 3)  # no
    assert g.query() == [True, False]
    g.delete("query", 1)
    assert g.query() == [True]
    print("OfflineQuery OK", g.query())


APPLICATION = """
- 在线难、离线易：把查询排序，只加「当前合法」边。
- 删边 ↔ 倒序加边；与 Kruskal 同构。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
