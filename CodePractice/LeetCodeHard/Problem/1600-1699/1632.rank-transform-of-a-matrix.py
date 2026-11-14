"""
LC 1632. Rank Transform of a Matrix — Hard
https://leetcode.com/problems/rank-transform-of-a-matrix/
Best: group by value + Union-Find rows/cols — O(mn log(mn))
"""

from typing import List
from collections import defaultdict


class Solution:
    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        m, n = len(matrix), len(matrix[0])
        rank = [0] * (m + n)
        ans = [[0] * n for _ in range(m)]
        vals = defaultdict(list)
        for i in range(m):
            for j in range(n):
                vals[matrix[i][j]].append((i, j))

        def find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for v in sorted(vals):
            parent = list(range(m + n))
            for i, j in vals[v]:
                pi, pj = find(parent, i), find(parent, j + m)
                parent[pi] = pj
            group_rank = defaultdict(int)
            for i, j in vals[v]:
                root = find(parent, i)
                group_rank[root] = max(group_rank[root], max(rank[i], rank[j + m]) + 1)
            for i, j in vals[v]:
                r = group_rank[find(parent, i)]
                ans[i][j] = r
                rank[i] = r
                rank[j + m] = r
        return ans


def _test():
    s = Solution()
    assert s.matrixRankTransform([[1, 2], [3, 4]]) == [[1, 2], [2, 3]]
    assert s.matrixRankTransform([[7, 7], [7, 7]]) == [[1, 1], [1, 1]]
    assert s.matrixRankTransform([[20, -21, 14], [-19, 4, 19], [22, -47, 24], [-19, 4, 19]]) == [
        [4, 2, 3],
        [1, 3, 4],
        [5, 1, 6],
        [1, 3, 4],
    ]
    print("LC 1632 OK")


if __name__ == "__main__":
    _test()
