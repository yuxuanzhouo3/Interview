"""
LC 329. Longest Increasing Path in a Matrix — Hard
Best: Memo DFS — O(mn)
"""

from typing import List
from functools import cache


class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])

        @cache
        def dfs(i, j):
            best = 1
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                    best = max(best, 1 + dfs(ni, nj))
            return best

        return max(dfs(i, j) for i in range(m) for j in range(n))


def _test():
    s = Solution()
    assert s.longestIncreasingPath([[9, 9, 4], [6, 6, 8], [2, 1, 1]]) == 4
    assert s.longestIncreasingPath([[3, 4, 5], [3, 2, 6], [2, 2, 1]]) == 4
    print("LC 329 OK")


if __name__ == "__main__":
    _test()
