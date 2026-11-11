"""
AdvancedGraph / DAG DP（拓扑 + 最长/最短路）
Template: query(dist) / update / add / del
App: 关键路径 LC 329/787 思想；项目关键路径；必须无环。
来源: CodePractice/graph.py §6.2
"""

from collections import defaultdict, deque


class DAGDP:
    def __init__(self):
        self.adj = defaultdict(dict)  # u -> {v: w}
        self.nodes = set()

    def add(self, u, v=None, w=1):
        self.nodes.add(u)
        if v is None:
            return
        self.nodes.add(v)
        self.adj[u][v] = w

    def update(self, u, v, w):
        if v not in self.adj[u]:
            return False
        self.adj[u][v] = w
        return True

    def delete(self, u, v=None):
        if v is None:
            self.nodes.discard(u)
            self.adj.pop(u, None)
            for s in self.adj.values():
                s.pop(u, None)
            return
        self.adj[u].pop(v, None)

    def query(self, start, mode="longest"):
        return self.run(start, mode)

    def _topo(self):
        indeg = {n: 0 for n in self.nodes}
        for u in self.adj:
            for v in self.adj[u]:
                indeg[v] += 1
        q = deque([n for n, d in indeg.items() if d == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return order if len(order) == len(self.nodes) else []

    def run(self, start, mode="longest"):
        order = self._topo()
        if not order:
            return None  # cycle
        if mode == "longest":
            INIT, better = float("-inf"), max
        else:
            INIT, better = float("inf"), min
        dist = {n: INIT for n in self.nodes}
        dist[start] = 0
        for u in order:
            if dist[u] == INIT:
                continue
            for v, w in self.adj[u].items():
                dist[v] = better(dist[v], dist[u] + w)
        return dist


def _test():
    g = DAGDP()
    g.add(0, 1, 3)
    g.add(0, 2, 2)
    g.add(1, 3, 4)
    g.add(2, 3, 10)
    d = g.query(0, "longest")
    assert d[3] == 12
    assert g.update(2, 3, 1)
    assert g.query(0, "longest")[3] == 7   # 0-1-3 = 3+4
    assert g.query(0, "shortest")[3] == 3  # 0-2-3 = 2+1
    g.delete(1, 3)
    assert g.query(0, "longest")[3] == 3   # 仅 0-2-3
    print("DAGDP OK", g.query(0))


APPLICATION = """
- 先拓扑再按序松弛；最长路/最短路同一框架。
- 有环返回 None；关键路径、树形 DP 外的 DAG 路径问题。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
