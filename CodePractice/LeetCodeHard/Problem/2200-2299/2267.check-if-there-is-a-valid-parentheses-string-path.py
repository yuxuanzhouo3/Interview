"""
LC 2267. Check if There Is a Valid Parentheses String Path — Hard
https://leetcode.com/problems/check-if-there-is-a-valid-parentheses-string-path/
Best: DFS/DP on balance — O(mn(m+n))
"""

from typing import List
from functools import cache


class Solution:
    def hasValidPath(self, grid: List[List[str]]) -> bool:
        m, n = len(grid), len(grid[0])
        if (m + n - 1) % 2 or grid[0][0] == ")" or grid[m - 1][n - 1] == "(":
            return False

        @cache
        def dfs(i: int, j: int, bal: int) -> bool:
            bal += 1 if grid[i][j] == "(" else -1
            if bal < 0 or bal > m - i + n - j:
                return False
            if i == m - 1 and j == n - 1:
                return bal == 0
            for di, dj in ((0, 1), (1, 0)):
                x, y = i + di, j + dj
                if 0 <= x < m and 0 <= y < n and dfs(x, y, bal):
                    return True
            return False

        return dfs(0, 0, 0)


def _test():
    s = Solution()
    assert s.hasValidPath(
        [["(", "(", "("], [")", "(", ")"], ["(", "(", ")"], ["(", "(", ")"]]
    ) is True
    assert s.hasValidPath([[")", ")"], ["(", "("]]) is False
    print("LC 2267 OK")


if __name__ == "__main__":
    _test()
