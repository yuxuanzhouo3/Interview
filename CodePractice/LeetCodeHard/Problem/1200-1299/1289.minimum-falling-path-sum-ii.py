"""
LC 1289. Minimum Falling Path Sum II — Hard
https://leetcode.com/problems/minimum-falling-path-sum-ii/
Best: Track top-2 mins per row — O(n^2)
"""

from typing import List


class Solution:
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        n = len(grid)
        prev = grid[0][:]
        for r in range(1, n):
            # find two smallest in prev
            m1 = m2 = float("inf")
            i1 = -1
            for j, v in enumerate(prev):
                if v < m1:
                    m2, m1, i1 = m1, v, j
                elif v < m2:
                    m2 = v
            cur = [0] * n
            for j in range(n):
                cur[j] = grid[r][j] + (m2 if j == i1 else m1)
            prev = cur
        return min(prev)


def _test():
    s = Solution()
    assert s.minFallingPathSum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 13
    assert s.minFallingPathSum([[7]]) == 7
    print("LC 1289 OK")


if __name__ == "__main__":
    _test()
