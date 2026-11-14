"""
LC 2392. Build a Matrix With Conditions — Hard
https://leetcode.com/problems/build-a-matrix-with-conditions/
Best: Topo sort rows & cols independently — O(k^2 + e)
"""

from typing import List
from collections import defaultdict, deque


class Solution:
    def buildMatrix(self, k: int, rowConditions: List[List[int]], columnConditions: List[List[int]]) -> List[List[int]]:
        def topo(conds: List[List[int]]) -> List[int]:
            g = defaultdict(list)
            indeg = [0] * (k + 1)
            for a, b in conds:
                g[a].append(b)
                indeg[b] += 1
            q = deque(i for i in range(1, k + 1) if indeg[i] == 0)
            order = []
            while q:
                u = q.popleft()
                order.append(u)
                for v in g[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
            return order if len(order) == k else []

        rows = topo(rowConditions)
        cols = topo(columnConditions)
        if not rows or not cols:
            return []
        pos_r = {x: i for i, x in enumerate(rows)}
        pos_c = {x: i for i, x in enumerate(cols)}
        mat = [[0] * k for _ in range(k)]
        for x in range(1, k + 1):
            mat[pos_r[x]][pos_c[x]] = x
        return mat


def _validate(mat, k, rowConditions, columnConditions):
    if not mat:
        return False
    pos = {}
    for i in range(k):
        for j in range(k):
            if mat[i][j]:
                pos[mat[i][j]] = (i, j)
    if set(pos) != set(range(1, k + 1)):
        return False
    for a, b in rowConditions:
        if pos[a][0] >= pos[b][0]:
            return False
    for a, b in columnConditions:
        if pos[a][1] >= pos[b][1]:
            return False
    return True


def _test():
    s = Solution()
    r1 = s.buildMatrix(3, [[1, 2], [3, 2]], [[2, 1], [3, 2]])
    assert _validate(r1, 3, [[1, 2], [3, 2]], [[2, 1], [3, 2]])
    assert s.buildMatrix(3, [[1, 2], [2, 3], [3, 1], [2, 3]], [[2, 1]]) == []
    print("LC 2392 OK")


if __name__ == "__main__":
    _test()
