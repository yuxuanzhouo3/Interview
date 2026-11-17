"""
LC 879. Profitable Schemes — Hard
Best: DP knapsack (members, profit) — O(n*G*P)
"""

from typing import List

MOD = 10**9 + 7


class Solution:
    def profitableSchemes(self, n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
        # dp[g][p] = ways using g members with profit >=p capped at minProfit
        P = minProfit
        dp = [[0] * (P + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for g, p in zip(group, profit):
            ndp = [row[:] for row in dp]
            for members in range(n - g + 1):
                for pr in range(P + 1):
                    v = dp[members][pr]
                    if not v:
                        continue
                    np_ = min(P, pr + p)
                    ndp[members + g][np_] = (ndp[members + g][np_] + v) % MOD
            dp = ndp
        return sum(dp[m][P] for m in range(n + 1)) % MOD


def _test():
    s = Solution()
    assert s.profitableSchemes(5, 3, [2, 2], [2, 3]) == 2
    assert s.profitableSchemes(10, 5, [2, 3, 5], [6, 7, 8]) == 7
    print("LC 879 OK")


if __name__ == "__main__":
    _test()
