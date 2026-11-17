"""
LC 1568. Minimum Number of Days to Disconnect Island — Hard
https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/
Best: Articulation / at most 2 — O((mn)^2)
"""

from typing import List


class Solution:
    def minDays(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        def islands() -> int:
            seen = [[False] * n for _ in range(m)]
            cnt = 0

            def dfs(x: int, y: int) -> None:
                seen[x][y] = True
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] and not seen[nx][ny]:
                        dfs(nx, ny)

            for i in range(m):
                for j in range(n):
                    if grid[i][j] and not seen[i][j]:
                        cnt += 1
                        dfs(i, j)
            return cnt

        if islands() != 1:
            return 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    grid[i][j] = 0
                    if islands() != 1:
                        return 1
                    grid[i][j] = 1
        return 2


def _test():
    s = Solution()
    assert s.minDays([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]) == 2
    assert s.minDays([[1, 1]]) == 2
    assert s.minDays([[1, 0, 1, 0]]) == 0
    print("LC 1568 OK")


if __name__ == "__main__":
    _test()
