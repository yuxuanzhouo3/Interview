"""
LC 2328. Number of Increasing Paths in a Grid — Hard
https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/
Best: DFS + memoization — O(mn)
"""

from typing import List
from functools import lru_cache


class Solution:
    def countPaths(self, grid: List[List[int]]) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        @lru_cache(None)
        def dfs(i: int, j: int) -> int:
            res = 1
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] > grid[i][j]:
                    res = (res + dfs(ni, nj)) % MOD
            return res

        return sum(dfs(i, j) for i in range(m) for j in range(n)) % MOD


def _test():
    s = Solution()
    assert s.countPaths([[1, 1], [3, 4]]) == 8
    assert s.countPaths([[1], [2]]) == 3
    print("LC 2328 OK")


if __name__ == "__main__":
    _test()
