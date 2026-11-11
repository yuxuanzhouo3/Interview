"""
AdvancedGraph / Cycle Detection（有向 / 无向环）
Template: query(has_cycle) / update / add / del
App: 死锁检测、课程表 LC 207、冗余连接 LC 684。
来源: CodePractice/graph.py §7
"""

from collections import defaultdict


class CycleGraph:
    def __init__(self, directed=False):
        self.directed = directed
        self.adj = defaultdict(set)
        self.nodes = set()

    def add(self, u, v=None):
        self.nodes.add(u)
        if v is None:
            return
        self.nodes.add(v)
        self.adj[u].add(v)
        if not self.directed:
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
            self.adj.pop(u, None)
            for s in self.adj.values():
                s.discard(u)
            return
        self.adj[u].discard(v)
        if not self.directed:
            self.adj[v].discard(u)

    def query(self):
        return self.has_cycle()

    def has_cycle(self):
        return (
            self._directed() if self.directed else self._undirected()
        )

    def _undirected(self):
        seen = set()

        def dfs(u, parent):
            seen.add(u)
            for v in self.adj[u]:
                if v not in seen:
                    if dfs(v, u):
                        return True
                elif v != parent:
                    return True
            return False

        for n in self.nodes:
            if n not in seen and dfs(n, None):
                return True
        return False

    def _directed(self):
        # 0 unvisited 1 visiting 2 done
        state = {}

        def dfs(u):
            state[u] = 1
            for v in self.adj[u]:
                if state.get(v) == 1:
                    return True
                if state.get(v, 0) == 0 and dfs(v):
                    return True
            state[u] = 2
            return False

        for n in self.nodes:
            if state.get(n, 0) == 0 and dfs(n):
                return True
        return False


def _test():
    u = CycleGraph(directed=False)
    u.add(0, 1)
    u.add(1, 2)
    assert not u.query()
    u.add(2, 0)
    assert u.query()
    u.delete(2, 0)
    assert not u.query()

    d = CycleGraph(directed=True)
    d.add(0, 1)
    d.add(1, 2)
    assert not d.query()
    d.add(2, 0)
    assert d.query()
    assert d.update(2, 0, 3)
    assert not d.query()
    print("Cycle OK")


APPLICATION = """
- 无向: DFS 父节点 / DSU 加边失败即环。
- 有向: 三色 DFS 或拓扑长度 < |V|。
- LC 207 Course Schedule；684 Redundant Connection。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
