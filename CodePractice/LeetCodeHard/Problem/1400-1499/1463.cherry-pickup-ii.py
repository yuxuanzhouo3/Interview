"""
LC 1463. Cherry Pickup II — Hard
https://leetcode.com/problems/cherry-pickup-ii/
Best: DP two robots same row — O(m n^2)
"""

from typing import List
from functools import cache


class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        @cache
        def dp(r, c1, c2):
            if c1 < 0 or c1 >= n or c2 < 0 or c2 >= n:
                return float("-inf")
            cherries = grid[r][c1] + (0 if c1 == c2 else grid[r][c2])
            if r == m - 1:
                return cherries
            best = float("-inf")
            for d1 in (-1, 0, 1):
                for d2 in (-1, 0, 1):
                    best = max(best, dp(r + 1, c1 + d1, c2 + d2))
            return cherries + best

        return dp(0, 0, n - 1)


def _test():
    s = Solution()
    assert s.cherryPickup([[3, 1, 1], [2, 5, 1], [1, 5, 5], [2, 1, 1]]) == 24
    assert s.cherryPickup([[1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 3, 0], [2, 0, 9, 0, 0, 0, 0], [0, 3, 0, 5, 4, 0, 0], [1, 0, 2, 3, 0, 0, 6]]) == 28
    print("LC 1463 OK")


if __name__ == "__main__":
    _test()
