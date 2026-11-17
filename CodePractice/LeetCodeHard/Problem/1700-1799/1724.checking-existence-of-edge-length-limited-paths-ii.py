"""
LC 1724. Checking Existence of Edge Length Limited Paths II — Hard
Best: Persistent DSU offline by weight — O((n+q) α)
Note: online version; build Kruskal tree / persistent UF
"""

from typing import List


class DistanceLimitedPathsExist:
    def __init__(self, n: int, edgeList: List[List[int]]):
        self.edges = sorted(edgeList, key=lambda e: e[2])
        self.n = n
        # snapshot DSU after each edge
        parent = list(range(n))

        def find(x, p):
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x

        self.snaps = []  # (weight, parent_copy) sparse: store versions
        self.weights = []
        self.parents = []
        p = list(range(n))
        for u, v, w in self.edges:
            ru, rv = find(u, p), find(v, p)
            if ru != rv:
                p[ru] = rv
            self.weights.append(w)
            self.parents.append(p[:])

    def query(self, p: int, q: int, limit: int) -> bool:
        # binary search last edge with weight < limit
        lo, hi = -1, len(self.weights) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.weights[mid] < limit:
                lo = mid
            else:
                hi = mid - 1
        if lo < 0:
            return False
        parent = self.parents[lo]

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        return find(p) == find(q)


def _test():
    d = DistanceLimitedPathsExist(6, [[0, 2, 4], [0, 3, 2], [1, 2, 3], [2, 3, 1], [4, 5, 5]])
    assert d.query(2, 3, 2) is True
    assert d.query(1, 3, 3) is False
    assert d.query(2, 0, 5) is True
    print("LC 1724 OK")


if __name__ == "__main__":
    _test()
