"""
LC 1579. Remove Max Number of Edges to Keep Graph Fully Traversable — Hard
https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/
Best: Union-Find prefer type-3 then Alice/Bob — O(E α(n))
"""

from typing import List


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n + 1))
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
        self.p[rb] = ra
        self.comp -= 1
        return True


class Solution:
    def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
        ua, ub = UnionFind(n), UnionFind(n)
        used = 0
        for t, u, v in edges:
            if t == 3:
                a = ua.union(u, v)
                b = ub.union(u, v)
                if a or b:
                    used += 1
        for t, u, v in edges:
            if t == 1 and ua.union(u, v):
                used += 1
            elif t == 2 and ub.union(u, v):
                used += 1
        if ua.comp > 1 or ub.comp > 1:
            return -1
        return len(edges) - used


def _test():
    s = Solution()
    assert s.maxNumEdgesToRemove(4, [[3, 1, 2], [3, 2, 3], [1, 1, 3], [1, 2, 4], [1, 1, 2], [2, 3, 4]]) == 2
    assert s.maxNumEdgesToRemove(4, [[3, 1, 2], [3, 2, 3], [1, 1, 4], [2, 1, 4]]) == 0
    assert s.maxNumEdgesToRemove(4, [[3, 2, 3], [1, 1, 2], [2, 3, 4]]) == -1
    print("LC 1579 OK")


if __name__ == "__main__":
    _test()
