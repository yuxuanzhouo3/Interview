"""
LC 2713. Maximum Strictly Increasing Cells in a Matrix — Hard
https://leetcode.com/problems/maximum-strictly-increasing-cells-in-a-matrix/
Best: Sort values + row/col DP — O(mn log(mn))
"""

from typing import List
from collections import defaultdict


class Solution:
    def maxIncreasingCells(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        pos = defaultdict(list)
        for i in range(m):
            for j in range(n):
                pos[mat[i][j]].append((i, j))
        row = [0] * m
        col = [0] * n
        ans = 0
        for val in sorted(pos):
            tmp = []
            for i, j in pos[val]:
                best = max(row[i], col[j]) + 1
                tmp.append((i, j, best))
                ans = max(ans, best)
            for i, j, best in tmp:
                row[i] = max(row[i], best)
                col[j] = max(col[j], best)
        return ans


def _test():
    s = Solution()
    assert s.maxIncreasingCells([[3, 1], [3, 4]]) == 2
    assert s.maxIncreasingCells([[1, 1], [1, 1]]) == 1
    assert s.maxIncreasingCells([[3, 1, 6], [-9, 5, 7]]) == 4
    print("LC 2713 OK")


if __name__ == "__main__":
    _test()

