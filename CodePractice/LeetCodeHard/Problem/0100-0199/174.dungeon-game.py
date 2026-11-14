"""
LC 174. Dungeon Game — Hard
Best: DP from bottom-right — min HP needed
"""

from typing import List


class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m, n = len(dungeon), len(dungeon[0])
        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        dp[m][n - 1] = dp[m - 1][n] = 1
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                need = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]
                dp[i][j] = 1 if need <= 0 else need
        return dp[0][0]


def _test():
    s = Solution()
    assert s.calculateMinimumHP([[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]) == 7
    print("LC 174 OK")


if __name__ == "__main__":
    _test()
