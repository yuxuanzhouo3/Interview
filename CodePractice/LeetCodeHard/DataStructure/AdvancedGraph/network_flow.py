"""
AdvancedGraph / Network Flow（网络流 · Dinic）
Template: query(max_flow) / update(cap) / add(edge) / del(edge)
App: 最大流/最小割；二分匹配也可建流网；LC 785 变体、吞吐上限。
来源精简: CodePractice/graph.py §8 Dinic
"""

from collections import deque


class Dinic:
    class Edge:
        __slots__ = ("to", "rev", "cap")

        def __init__(self, to, rev, cap):
            self.to = to
            self.rev = rev
            self.cap = cap

    def __init__(self, n=0):
        self.n = n
        self.graph = [[] for _ in range(n)]
        # 记录 (u,v) -> forward edge index，便于 update/delete
        self._fwd = {}

    def add(self, u, v=None, cap=0):
        """add 节点扩容；或 add(u,v,cap) 加容量边（含反向边）。"""
        if v is None:
            # 扩到至少含节点 u
            while self.n <= u:
                self.graph.append([])
                self.n += 1
            return
        need = max(u, v) + 1
        while self.n < need:
            self.graph.append([])
            self.n += 1
        if (u, v) in self._fwd:
            # 已有边则累加容量
            ei = self._fwd[(u, v)]
            self.graph[u][ei].cap += cap
            return
        forward = self.Edge(v, len(self.graph[v]), cap)
        backward = self.Edge(u, len(self.graph[u]), 0)
        self._fwd[(u, v)] = len(self.graph[u])
        self.graph[u].append(forward)
        self.graph[v].append(backward)

    def update(self, u, v, cap):
        """将 u→v 正向剩余容量设为 cap（重置该边容量，反向清零相关）。"""
        if (u, v) not in self._fwd:
            return False
        ei = self._fwd[(u, v)]
        e = self.graph[u][ei]
        rev = self.graph[v][e.rev]
        # 重置为新容量（教学版：丢弃已推流量状态，调用方应在跑流前 update）
        e.cap = cap
        rev.cap = 0
        return True

    def delete(self, u, v=None):
        """delete(u,v) 将容量置 0；delete(u) 不物理删点（流图少用）。"""
        if v is None:
            return
        self.update(u, v, 0)

    def query(self, s, t):
        """最大流 s→t。"""
        return self.max_flow(s, t)

    def bfs(self, s, t):
        self.level = [-1] * self.n
        q = deque([s])
        self.level[s] = 0
        while q:
            u = q.popleft()
            for e in self.graph[u]:
                if e.cap > 0 and self.level[e.to] == -1:
                    self.level[e.to] = self.level[u] + 1
                    q.append(e.to)
        return self.level[t] != -1

    def dfs(self, u, t, pushed):
        if pushed == 0:
            return 0
        if u == t:
            return pushed
        while self.it[u] < len(self.graph[u]):
            e = self.graph[u][self.it[u]]
            if e.cap > 0 and self.level[e.to] == self.level[u] + 1:
                flow = self.dfs(e.to, t, min(pushed, e.cap))
                if flow:
                    e.cap -= flow
                    self.graph[e.to][e.rev].cap += flow
                    return flow
            self.it[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0
        INF = 10**18
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                pushed = self.dfs(s, t, INF)
                if pushed == 0:
                    break
                flow += pushed
        return flow


def _test():
    # 经典: s=0 → 1,2 → t=3
    g = Dinic(4)
    g.add(0, 1, 10)
    g.add(0, 2, 10)
    g.add(1, 2, 2)
    g.add(1, 3, 8)
    g.add(2, 3, 10)
    assert g.query(0, 3) == 18

    # 新图：update / delete
    h = Dinic()
    h.add(0)  # ensure node
    h.add(0, 1, 5)
    h.add(1, 2, 5)
    assert h.query(0, 2) == 5
    # 重建再测 update（Dinic 跑完会改 residual）
    h2 = Dinic(3)
    h2.add(0, 1, 5)
    h2.add(1, 2, 5)
    assert h2.update(0, 1, 3)
    assert h2.query(0, 2) == 3
    h3 = Dinic(3)
    h3.add(0, 1, 5)
    h3.add(1, 2, 5)
    h3.delete(0, 1)
    assert h3.query(0, 2) == 0
    print("NetworkFlow OK", 18)


APPLICATION = """
- 最大流 = 最小割：带宽、货运、项目人选上限。
- 二分图匹配可建：源→左部→右部→汇，边权 1。
- Edmonds-Karp = BFS 增广 O(VE²)；Dinic 分层 + 多路增广，稠密/竞赛常用。
- 最小费用最大流：在残差上 SPFA/Dijkstra 找增广。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
