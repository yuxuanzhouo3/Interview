"""
LC 2312. Selling Pieces of Wood — Hard
https://leetcode.com/problems/selling-pieces-of-wood/
Best: DP on rectangle cuts — O(m^2 n^2 + prices)
"""

from typing import List


class Solution:
    def sellingWood(self, m: int, n: int, prices: List[List[int]]) -> int:
        pr = [[0] * (n + 1) for _ in range(m + 1)]
        for h, w, p in prices:
            pr[h][w] = max(pr[h][w], p)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for h in range(1, m + 1):
            for w in range(1, n + 1):
                best = pr[h][w]
                for i in range(1, h // 2 + 1):
                    best = max(best, dp[i][w] + dp[h - i][w])
                for j in range(1, w // 2 + 1):
                    best = max(best, dp[h][j] + dp[h][w - j])
                dp[h][w] = best
        return dp[m][n]


def _test():
    s = Solution()
    assert s.sellingWood(3, 5, [[1, 4, 2], [2, 2, 7], [2, 1, 3]]) == 19
    assert s.sellingWood(4, 6, [[3, 2, 10], [1, 4, 2], [4, 1, 3]]) == 32
    print("LC 2312 OK")


if __name__ == "__main__":
    _test()
