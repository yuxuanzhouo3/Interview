"""
AdvancedGraph / SCC（Tarjan 强连通分量）
Template: query(sccs) / update(edge) / add / del
App: 缩点、2-SAT、LC 1192 相关连通；有向图强连通分解。
来源: CodePractice/graph.py §3.3 Tarjan
"""

from collections import defaultdict


class SCCGraph:
    def __init__(self):
        self.adj = defaultdict(set)
        self.nodes = set()

    def add(self, u, v=None):
        self.nodes.add(u)
        if v is None:
            return
        self.nodes.add(v)
        self.adj[u].add(v)

    def update(self, u, old_v, new_v):
        if old_v not in self.adj[u]:
            return False
        self.adj[u].remove(old_v)
        self.adj[u].add(new_v)
        self.nodes.add(new_v)
        return True

    def delete(self, u, v=None):
        if v is None:
            self.nodes.discard(u)
            self.adj.pop(u, None)
            for s in self.adj.values():
                s.discard(u)
            return
        self.adj[u].discard(v)

    def query(self):
        """Tarjan SCC 列表。"""
        return self.tarjan()

    def tarjan(self):
        ids = {n: i for i, n in enumerate(sorted(self.nodes))}
        rev = {i: n for n, i in ids.items()}
        n = len(ids)
        g = [[] for _ in range(n)]
        for u in self.adj:
            for v in self.adj[u]:
                if u in ids and v in ids:
                    g[ids[u]].append(ids[v])

        dfn = [-1] * n
        low = [0] * n
        stack, in_stack = [], [False] * n
        timestamp, sccs = 0, []

        def dfs(u):
            nonlocal timestamp
            dfn[u] = low[u] = timestamp
            timestamp += 1
            stack.append(u)
            in_stack[u] = True
            for v in g[u]:
                if dfn[v] == -1:
                    dfs(v)
                    low[u] = min(low[u], low[v])
                elif in_stack[v]:
                    low[u] = min(low[u], dfn[v])
            if low[u] == dfn[u]:
                comp = []
                while True:
                    x = stack.pop()
                    in_stack[x] = False
                    comp.append(rev[x])
                    if x == u:
                        break
                sccs.append(comp)

        for u in range(n):
            if dfn[u] == -1:
                dfs(u)
        return sccs


def _test():
    g = SCCGraph()
    # 0→1→2→0 一个 SCC；3→4；4→3
    for e in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (2, 3)]:
        g.add(*e)
    sccs = g.query()
    assert len(sccs) == 2
    sizes = sorted(len(c) for c in sccs)
    assert sizes == [2, 3]
    assert g.update(2, 3, 4)
    g.delete(4, 3)
    print("SCC OK", g.query())


APPLICATION = """
- 缩点成 DAG 再 DP / 拓扑；2-SAT 建边求 SCC。
- Kosaraju = 两遍 DFS；Tarjan 一遍 + 栈。
- LC: 强连通相关、Critical Connections 用桥（见 bridge_cut）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
