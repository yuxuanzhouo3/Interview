"""
LC 1692. Count Ways to Distribute Candies — Hard
https://leetcode.com/problems/count-ways-to-distribute-candies/
Best: Stirling 2nd kind DP — O(nk)
"""


class Solution:
    def waysToDistribute(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = (dp[i - 1][j] * j + dp[i - 1][j - 1]) % MOD
        return dp[n][k]


def _test():
    s = Solution()
    assert s.waysToDistribute(3, 2) == 3
    assert s.waysToDistribute(4, 2) == 7
    assert s.waysToDistribute(20, 5) == 206085257
    print("LC 1692 OK")


if __name__ == "__main__":
    _test()
