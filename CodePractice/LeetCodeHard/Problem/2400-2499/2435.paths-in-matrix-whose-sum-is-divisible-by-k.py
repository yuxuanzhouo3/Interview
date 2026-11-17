"""
LC 2435. Paths in Matrix Whose Sum Is Divisible by K — Hard
https://leetcode.com/problems/paths-in-matrix-whose-sum-is-divisible-by-k/
Best: DP on (i,j,mod) — O(mnk)
"""

from typing import List


class Solution:
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        dp = [[[0] * k for _ in range(n)] for _ in range(m)]
        dp[0][0][grid[0][0] % k] = 1
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                for r in range(k):
                    ways = 0
                    if i:
                        ways += dp[i - 1][j][r]
                    if j:
                        ways += dp[i][j - 1][r]
                    dp[i][j][(r + grid[i][j]) % k] = (dp[i][j][(r + grid[i][j]) % k] + ways) % MOD
        return dp[m - 1][n - 1][0]


def _test():
    s = Solution()
    assert s.numberOfPaths([[5, 2, 4], [3, 0, 5], [0, 7, 2]], 3) == 2
    assert s.numberOfPaths([[0, 0]], 5) == 1
    assert s.numberOfPaths([[7, 3, 4, 9], [2, 3, 6, 2], [2, 3, 7, 0]], 1) == 10
    print("LC 2435 OK")


if __name__ == "__main__":
    _test()
