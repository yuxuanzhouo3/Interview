"""
LC 980. Unique Paths III — Hard
https://leetcode.com/problems/unique-paths-iii/
Best: Backtracking / bit DP — O(3^{empty})
"""

from typing import List


class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        empty = 0
        sr = sc = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] != -1:
                    empty += 1
                if grid[i][j] == 1:
                    sr, sc = i, j
        self.ans = 0

        def dfs(r, c, left):
            if grid[r][c] == 2:
                if left == 1:
                    self.ans += 1
                return
            tmp = grid[r][c]
            grid[r][c] = -1
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] != -1:
                    dfs(nr, nc, left - 1)
            grid[r][c] = tmp

        dfs(sr, sc, empty)
        return self.ans


def _test():
    s = Solution()
    assert s.uniquePathsIII([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, -1]]) == 2
    assert s.uniquePathsIII([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]) == 4
    assert s.uniquePathsIII([[0, 1], [2, 0]]) == 0
    print("LC 980 OK")


if __name__ == "__main__":
    _test()
