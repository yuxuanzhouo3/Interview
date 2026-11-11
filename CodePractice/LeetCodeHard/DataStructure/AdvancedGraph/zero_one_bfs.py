"""
AdvancedGraph / 0-1 BFS
Template: query(dist) / update(w) / add / del
App: 边权仅 0/1；网格改代价、门开关；比 Dijkstra 常数更好 O(V+E)。
来源: CodePractice/graph.py §4.2
"""

from collections import defaultdict, deque


class ZeroOneBFS:
    def __init__(self):
        self.adj = defaultdict(dict)  # u -> {v: w in {0,1}}

    def add(self, u, v, w=1):
        assert w in (0, 1)
        self.adj[u][v] = w

    def update(self, u, v, w):
        if v not in self.adj[u]:
            return False
        assert w in (0, 1)
        self.adj[u][v] = w
        return True

    def delete(self, u, v):
        self.adj[u].pop(v, None)

    def query(self, src, dst=None):
        dist = self.bfs(src)
        return dist if dst is None else dist.get(dst, float("inf"))

    def bfs(self, start):
        INF = float("inf")
        dist = {start: 0}
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v, w in self.adj[u].items():
                nd = dist[u] + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    if w == 0:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        return dist


def _test():
    g = ZeroOneBFS()
    g.add(0, 1, 0)
    g.add(1, 2, 1)
    g.add(0, 2, 1)
    assert g.query(0, 2) == 1
    assert g.update(0, 2, 0)
    assert g.query(0, 2) == 0
    g.delete(0, 2)
    assert g.query(0, 2) == 1
    print("01BFS OK", g.query(0))


APPLICATION = """
- 边权 ∈ {0,1} 用双端队列：0 边插队头，1 边队尾。
- 网格「翻墙代价 1 / 走路 0」类题；勿对一般权值用。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
