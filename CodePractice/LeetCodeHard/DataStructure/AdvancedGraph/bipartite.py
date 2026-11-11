"""
AdvancedGraph / Bipartite（二分图）
Template: query(is_bipartite / matching) / update(edge) / add / del
App: 染色判定 LC 785；二分图最大匹配 → 任务分配、LC 785/886/785 变体。
来源精简: CodePractice/graph.py §9 Matching
"""

from collections import defaultdict, deque


class BipartiteGraph:
    """
    无向图；支持二分图判定（2-coloring）+ 左→右最大匹配（DFS 增广）。
    匹配约定：节点可标 'L'/'R' 侧；未指定时用 color 划分后按两侧匹配。
    """

    def __init__(self):
        self.adj = defaultdict(set)
        self.nodes = set()
        self.side = {}  # node -> 'L' | 'R'（可选，手动指定分区）

    def add(self, u, v=None, side=None):
        """add(u) 加点；add(u,v) 加无向边；side 可标 'L'/'R'。"""
        self.nodes.add(u)
        if side is not None:
            self.side[u] = side
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
            self.side.pop(u, None)
            for nei in list(self.adj.get(u, ())):
                self.adj[nei].discard(u)
            self.adj.pop(u, None)
            return
        self.adj[u].discard(v)
        self.adj[v].discard(u)

    def query(self, mode="bipartite"):
        """
        mode='bipartite' -> (ok, color)
        mode='matching'  -> (size, match_right) 需已是二分图；左=side L 或 color0
        """
        if mode == "bipartite":
            return self.is_bipartite()
        if mode == "matching":
            return self.max_matching()
        raise ValueError(mode)

    def is_bipartite(self):
        """BFS 染色；返回 (True, color_dict) 或 (False, {})。"""
        color = {}
        for start in self.nodes:
            if start in color:
                continue
            color[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in color:
                        color[v] = color[u] ^ 1
                        q.append(v)
                    elif color[v] == color[u]:
                        return False, {}
        return True, color

    def max_matching(self):
        """
        DFS 增广路最大匹配（来自 graph.py bipartite_matching）。
        左部: side=='L'，否则用染色 color==0；右部另一侧。
        返回 (matching_size, {right: left})。
        """
        ok, color = self.is_bipartite()
        if not ok:
            return 0, {}

        if self.side:
            left = [n for n in self.nodes if self.side.get(n) == "L"]
            right = [n for n in self.nodes if self.side.get(n) == "R"]
        else:
            left = [n for n in self.nodes if color.get(n, 0) == 0]
            right = [n for n in self.nodes if color.get(n, 0) == 1]

        r_idx = {r: i for i, r in enumerate(right)}
        graph = [[] for _ in left]
        for i, u in enumerate(left):
            for v in self.adj[u]:
                if v in r_idx:
                    graph[i].append(r_idx[v])

        match_right = [-1] * len(right)

        def dfs(u, visited):
            for v in graph[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_right[v] == -1 or dfs(match_right[v], visited):
                    match_right[v] = u
                    return True
            return False

        matching = 0
        for u in range(len(left)):
            visited = [False] * len(right)
            if dfs(u, visited):
                matching += 1

        pairs = {
            right[ri]: left[match_right[ri]]
            for ri in range(len(right))
            if match_right[ri] != -1
        }
        return matching, pairs


def _test():
    g = BipartiteGraph()
    for e in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        g.add(*e)
    ok, color = g.query("bipartite")
    assert ok and len(color) == 4

    g.add(0, 2)  # 奇环
    assert not g.query("bipartite")[0]
    g.delete(0, 2)
    assert g.query("bipartite")[0]

    m = BipartiteGraph()
    for x, s in [(0, "L"), (1, "L"), (2, "R"), (3, "R")]:
        m.add(x, side=s)
    m.add(0, 2)
    m.add(1, 3)
    size, pairs = m.query("matching")
    assert size == 2 and pairs[2] == 0 and pairs[3] == 1

    assert m.update(1, 3, 2)  # 两边都连 2
    assert m.query("matching")[0] == 1
    m.delete(0, 2)
    assert m.query("matching")[0] == 1  # 仅 1-2
    print("Bipartite OK", size, pairs)


APPLICATION = """
- 判定: LC 785 Is Graph Bipartite；冲突图 2-着色。
- 匹配: 工位/任务分配、男女配对、最小路径覆盖(DAG→二分)。
- König: 二分图最大匹配 = 最小点覆盖；考试常考。
- 更大图用 Hopcroft-Karp；带权用 KM / min-cost flow。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
