"""
LC 1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree — Hard
https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
Best: Kruskal force / exclude / include — O(E^2 α(V))
"""

from typing import List


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
        self.comp = n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        self.comp -= 1
        return True


class Solution:
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        m = len(edges)
        es = [[u, v, w, i] for i, (u, v, w) in enumerate(edges)]
        es.sort(key=lambda x: x[2])

        def mst(forbid=-1, must=-1):
            uf = UnionFind(n)
            cost = 0
            if must >= 0:
                u, v, w, _ = es[must]
                uf.union(u, v)
                cost += w
            for i, (u, v, w, _) in enumerate(es):
                if i == forbid or i == must:
                    continue
                if uf.union(u, v):
                    cost += w
            return cost if uf.comp == 1 else float("inf")

        base = mst()
        critical, pseudo = [], []
        for i in range(m):
            if mst(forbid=i) > base:
                critical.append(es[i][3])
            elif mst(must=i) == base:
                pseudo.append(es[i][3])
        return [critical, pseudo]


def _test():
    s = Solution()
    assert s.findCriticalAndPseudoCriticalEdges(5, [[0, 1, 1], [1, 2, 1], [2, 3, 2], [0, 3, 2], [0, 4, 3], [3, 4, 3], [1, 4, 6]]) == [[0, 1], [2, 3, 4, 5]]
    assert s.findCriticalAndPseudoCriticalEdges(4, [[0, 1, 1], [1, 2, 1], [2, 3, 1], [0, 3, 1]]) == [[], [0, 1, 2, 3]]
    print("LC 1489 OK")


if __name__ == "__main__":
    _test()
