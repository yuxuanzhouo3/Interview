"""
AdvancedGraph / Bridge & Articulation（桥 / 割点 · Tarjan）
Template: query(bridges|cuts) / update / add / del
App: LC 1192 Critical Connections；网络单点/单边故障。
大纲: graph.py §3 Bridge / Articulation（补实现）
"""

from collections import defaultdict


class BridgeCutGraph:
    def __init__(self):
        self.adj = defaultdict(set)
        self.nodes = set()

    def add(self, u, v=None):
        self.nodes.add(u)
        if v is None:
            return
        self.nodes.add(v)
        self.adj[u].add(v)
        self.adj[v].add(u)

    def update(self, u, old_v, new_v):
        if old_v not in self.adj[u]:
            return False
        self.delete(u, old_v)
        self.add(u, new_v)
        return True

    def delete(self, u, v=None):
        if v is None:
            self.nodes.discard(u)
            for nei in list(self.adj.get(u, ())):
                self.adj[nei].discard(u)
            self.adj.pop(u, None)
            return
        self.adj[u].discard(v)
        self.adj[v].discard(u)

    def query(self, mode="bridges"):
        if mode == "bridges":
            return self.bridges()
        if mode == "cuts":
            return self.articulation_points()
        raise ValueError(mode)

    def _tarjan_meta(self):
        dfn, low, parent = {}, {}, {}
        time = 0
        bridges, cuts = [], set()
        children = defaultdict(int)

        def dfs(u):
            nonlocal time
            dfn[u] = low[u] = time
            time += 1
            for v in self.adj[u]:
                if v not in dfn:
                    parent[v] = u
                    children[u] += 1
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > dfn[u]:
                        bridges.append(tuple(sorted((u, v))))
                    if parent.get(u) is not None and low[v] >= dfn[u]:
                        cuts.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], dfn[v])
            if parent.get(u) is None and children[u] > 1:
                cuts.add(u)

        for n in self.nodes:
            if n not in dfn:
                parent[n] = None
                dfs(n)
        return bridges, cuts

    def bridges(self):
        return self._tarjan_meta()[0]

    def articulation_points(self):
        return self._tarjan_meta()[1]


def _test():
    g = BridgeCutGraph()
    # 0-1-2 成环，2-3 是桥，3 割点一侧
    for e in [(0, 1), (1, 2), (2, 0), (2, 3)]:
        g.add(*e)
    br = g.query("bridges")
    assert (2, 3) in br or (3, 2) in [tuple(sorted(x)) for x in br]
    assert 2 in g.query("cuts")
    assert g.update(2, 3, 4)
    g.delete(1, 2)
    print("BridgeCut OK", g.query("bridges"), g.query("cuts"))


APPLICATION = """
- 桥: 删边增加连通分量；割点: 删点增加连通分量。
- LC 1192 Critical Connections in a Network。
- 无向图 Tarjan：low[v] > dfn[u] → 桥；low[v] >= dfn[u] → 非根割点。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
