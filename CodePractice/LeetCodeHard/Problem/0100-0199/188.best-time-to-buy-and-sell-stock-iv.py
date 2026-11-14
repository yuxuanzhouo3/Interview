"""
LC 188. Best Time to Buy and Sell Stock IV — Hard
Best: DP k transactions; if k>=n/2 unlimited
"""

from typing import List


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        if not prices or k == 0:
            return 0
        if k >= n // 2:
            return sum(max(prices[i] - prices[i - 1], 0) for i in range(1, n))
        dp = [[0] * n for _ in range(k + 1)]
        for t in range(1, k + 1):
            max_diff = -prices[0]
            for i in range(1, n):
                dp[t][i] = max(dp[t][i - 1], prices[i] + max_diff)
                max_diff = max(max_diff, dp[t - 1][i] - prices[i])
        return dp[k][n - 1]


def _test():
    s = Solution()
    assert s.maxProfit(2, [2, 4, 1]) == 2
    assert s.maxProfit(2, [3, 2, 6, 5, 0, 3]) == 7
    print("LC 188 OK")


if __name__ == "__main__":
    _test()
