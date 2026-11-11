"""
4. 离线 DSU：按权/时间排序再合并
App: Kruskal 思想；LC 1697 思想。
"""


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def distance_limited_paths(n, edge_list, queries):
    """
    LC 1697: queries[i]=[p,q,limit]，是否存在路径上边权都 < limit。
    离线：边与询问按权排序，渐进 union。
    """
    edges = sorted(edge_list, key=lambda e: e[2])
    qi = sorted(range(len(queries)), key=lambda i: queries[i][2])
    dsu = DSU(n)
    ans = [False] * len(queries)
    j = 0
    for i in qi:
        p, q, limit = queries[i]
        while j < len(edges) and edges[j][2] < limit:
            u, v, _ = edges[j]
            dsu.union(u, v)
            j += 1
        ans[i] = dsu.find(p) == dsu.find(q)
    return ans


def mst_weight(n, edges):
    edges = sorted(edges, key=lambda e: e[2])
    dsu = DSU(n)
    w = cnt = 0
    for u, v, c in edges:
        if dsu.find(u) != dsu.find(v):
            dsu.union(u, v)
            w += c
            cnt += 1
            if cnt == n - 1:
                break
    return w if cnt == n - 1 else -1


def _test():
    # LC 1697 ex1
    edges = [[0, 1, 2], [1, 2, 4], [2, 0, 8], [1, 0, 5]]
    q = [[0, 1, 2], [0, 2, 5]]
    assert distance_limited_paths(3, edges, q) == [False, True]
    assert mst_weight(3, [[0, 1, 2], [1, 2, 4], [0, 2, 8]]) == 6
    print("dsu_offline OK", distance_limited_paths(3, edges, q))


APPLICATION = """
- 删边离线 ≡ 倒序加边；询问按阈值渐进激活边。
- Kruskal = 按权加边的离线连通。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
