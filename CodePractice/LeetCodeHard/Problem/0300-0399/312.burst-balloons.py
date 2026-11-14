"""
LC 312. Burst Balloons — Hard
Best: Interval DP — O(n^3)
"""

from typing import List


class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        a = [1] + nums + [1]
        n = len(a)
        dp = [[0] * n for _ in range(n)]
        for length in range(2, n):
            for l in range(0, n - length):
                r = l + length
                for k in range(l + 1, r):
                    dp[l][r] = max(
                        dp[l][r],
                        dp[l][k] + a[l] * a[k] * a[r] + dp[k][r],
                    )
        return dp[0][n - 1]


def _test():
    s = Solution()
    assert s.maxCoins([3, 1, 5, 8]) == 167
    assert s.maxCoins([1, 5]) == 10
    print("LC 312 OK")


if __name__ == "__main__":
    _test()
