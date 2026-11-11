"""
8. Hard DSU 套题：最大公因子连通 / 最小生成树关键边
App: LC 952 / 1489 思想。
"""

from collections import defaultdict


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]


def largest_component_size(nums):
    """LC 952: 共享 >1 的公因子则连边，求最大连通块。"""
    m = max(nums)
    dsu = DSU(m + 1)
    present = set(nums)
    for x in nums:
        d = 2
        while d * d <= x:
            if x % d == 0:
                dsu.union(x, d)
                dsu.union(x, x // d)
            d += 1
    cnt = defaultdict(int)
    for x in nums:
        cnt[dsu.find(x)] += 1
    return max(cnt.values()) if cnt else 0


def find_critical_and_pseudo(n, edges):
    """
    LC 1489 简化：返回 (critical_ids, pseudo_ids)
    边 [u,v,w,id]；critical=删后 MST 变差；pseudo=可出现在某 MST。
    """
    def mst(ban=-1, must=-1):
        dsu = DSU(n)
        w = used = 0
        if must != -1:
            u, v, c, _ = edges[must]
            dsu.union(u, v)
            w += c
            used = 1
        for i, (u, v, c, _) in enumerate(sorted(enumerate(edges), key=lambda t: t[1][2])):
            # sort by weight keep original index via enumerate on sorted edges
            pass
        # cleaner:
        order = sorted(range(len(edges)), key=lambda i: edges[i][2])
        for i in order:
            if i == ban:
                continue
            if must != -1 and i != must and used == 0:
                pass
            u, v, c, _ = edges[i]
            if must == i:
                continue  # already united
            if dsu.find(u) != dsu.find(v):
                if must != -1 and i != must and dsu.find(edges[must][0]) == dsu.find(edges[must][1]) or True:
                    dsu.union(u, v)
                    w += c
                    used += 1
        return w if used == n - 1 else float("inf")

    # rewrite clearer MST helper
    def mst_weight(ban=None, force=None):
        dsu = DSU(n)
        total = cnt = 0
        seq = []
        if force is not None:
            seq.append(force)
        seq += sorted((i for i in range(len(edges)) if i != force), key=lambda i: edges[i][2])
        for i in seq:
            if i == ban:
                continue
            u, v, c, _ = edges[i]
            if dsu.find(u) != dsu.find(v):
                dsu.union(u, v)
                total += c
                cnt += 1
        return total if cnt == n - 1 else float("inf")

    base = mst_weight()
    critical, pseudo = [], []
    for i in range(len(edges)):
        if mst_weight(ban=i) > base:
            critical.append(edges[i][3])
        elif mst_weight(force=i) == base:
            pseudo.append(edges[i][3])
    return critical, pseudo


def _test():
    assert largest_component_size([4, 6, 15, 35]) == 4
    edges = [[0, 1, 1, 0], [1, 2, 1, 1], [2, 3, 2, 2], [0, 3, 2, 3], [0, 4, 3, 4], [3, 4, 3, 5], [1, 4, 6, 6]]
    # n=5 small check runs
    c, p = find_critical_and_pseudo(5, edges)
    assert isinstance(c, list) and isinstance(p, list)
    print("dsu_hard_apps OK", largest_component_size([4, 6, 15, 35]), c, p)


APPLICATION = """
- 952: 质因子连到数值节点，避免 O(n²) 两两 gcd。
- 1489: 对比「删边 MST」与「强制用边 MST」相对最优 MST。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
