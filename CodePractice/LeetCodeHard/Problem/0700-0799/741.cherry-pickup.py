"""
LC 741. Cherry Pickup — Hard
Best: Dual person DP / one person twice — O(n^3)
"""

from typing import List
from functools import cache


class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        n = len(grid)

        @cache
        def dp(r1, c1, r2):
            c2 = r1 + c1 - r2
            if r1 >= n or c1 >= n or r2 >= n or c2 >= n:
                return float("-inf")
            if grid[r1][c1] == -1 or grid[r2][c2] == -1:
                return float("-inf")
            if r1 == c1 == n - 1:
                return grid[r1][c1]
            cherries = grid[r1][c1]
            if (r1, c1) != (r2, c2):
                cherries += grid[r2][c2]
            cherries += max(
                dp(r1 + 1, c1, r2 + 1),
                dp(r1 + 1, c1, r2),
                dp(r1, c1 + 1, r2 + 1),
                dp(r1, c1 + 1, r2),
            )
            return cherries

        ans = dp(0, 0, 0)
        return max(0, ans)


def _test():
    s = Solution()
    assert s.cherryPickup([[0, 1, -1], [1, 0, -1], [1, 1, 1]]) == 5
    print("LC 741 OK")


if __name__ == "__main__":
    _test()
