"""
LC 3989. Maximum Consistent Columns in a Grid — Hard
https://leetcode.com/problems/maximum-consistent-columns-in-a-grid/
Best: DP longest column subsequence with pairwise |diff|<=limit per row
"""


class Solution:
    def maxConsistentColumns(self, grid: list[list[int]], limit: int) -> int:
        m, n = len(grid), len(grid[0])
        dp = [1] * n
        for j in range(n):
            for i in range(j):
                ok = True
                for r in range(m):
                    if abs(grid[r][j] - grid[r][i]) > limit:
                        ok = False
                        break
                if ok:
                    dp[j] = max(dp[j], dp[i] + 1)
        return max(dp)


def _test():
    s = Solution()
    assert s.maxConsistentColumns([[-2, 0, 3]], 2) == 2
    assert s.maxConsistentColumns([[1, -1, 1], [2, 2, 2]], 1) == 2
    assert s.maxConsistentColumns([[-5, 5]], 9) == 1
    print("LC 3989 OK")


if __name__ == "__main__":
    _test()
