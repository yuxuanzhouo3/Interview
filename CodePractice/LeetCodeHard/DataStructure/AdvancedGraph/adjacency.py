"""
AdvancedGraph / Adjacency List Graph
Template: query / update(edge weight) / add / del
App: 通用图建模；社交、路由、依赖图。
"""

from collections import defaultdict, deque


class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.adj = defaultdict(dict)  # u -> {v: w}

    def add(self, u, v, w=1):
        self.adj[u][v] = w
        if not self.directed:
            self.adj[v][u] = w

    def update(self, u, v, w):
        if v not in self.adj.get(u, {}):
            return False
        self.adj[u][v] = w
        if not self.directed:
            self.adj[v][u] = w
        return True

    def query(self, u, v=None):
        """query(u) -> neighbors; query(u,v) -> weight or None"""
        if v is None:
            return dict(self.adj.get(u, {}))
        return self.adj.get(u, {}).get(v)

    def delete(self, u, v=None):
        """delete edge u-v；或 delete(u) 删顶点。"""
        if v is None:
            for nei in list(self.adj.get(u, {})):
                self.adj[nei].pop(u, None)
            self.adj.pop(u, None)
            return
        self.adj.get(u, {}).pop(v, None)
        if not self.directed:
            self.adj.get(v, {}).pop(u, None)

    def bfs(self, start):
        seen, order = {start}, []
        q = deque([start])
        while q:
            x = q.popleft()
            order.append(x)
            for y in self.adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        return order


def _test():
    g = Graph()
    g.add(1, 2, 3)
    g.add(2, 3, 4)
    assert g.query(1, 2) == 3
    assert g.update(1, 2, 9)
    assert g.query(1)[2] == 9
    g.delete(1, 2)
    assert g.query(1, 2) is None
    g.delete(2)  # clear remnant node/edges
    g.add(1, 2)
    g.add(2, 4)
    assert g.bfs(1) == [1, 2, 4]
    print("Graph OK", g.bfs(1))


APPLICATION = """
- 业务图: 好友关系、推荐边、微服务依赖、编译依赖。
- 矩阵适合稠密小图；海量稀疏用邻接表（本模板）。
- 进阶: Dijkstra / SPFA / Floyd / 拓扑 → 见同目录其他文件。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
