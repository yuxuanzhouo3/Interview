"""
AdvancedGraph / Topological Sort (Kahn)
Template: query(order) / update(edge) / add / del
App: 课程表、编译依赖、任务流水线 LC 210。
"""

from collections import defaultdict, deque


class DAG:
    def __init__(self):
        self.adj = defaultdict(set)
        self.nodes = set()

    def add(self, u, v=None):
        """add node u；或 add edge u→v。"""
        self.nodes.add(u)
        if v is not None:
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
        """拓扑序；有环返回 []。"""
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


def _test():
    g = DAG()
    g.add(0, 1)
    g.add(0, 2)
    g.add(1, 3)
    g.add(2, 3)
    assert g.query() in ([0, 1, 2, 3], [0, 2, 1, 3])
    assert g.update(0, 2, 3)
    g.delete(1, 3)
    print("Topo OK", g.query())


APPLICATION = """
- 包管理依赖、CI pipeline、课程表。
- 有环检测：拓扑长度 < |V|。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
