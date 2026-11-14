"""
LC 2188. Minimum Time to Finish the Race — Hard
https://leetcode.com/problems/minimum-time-to-finish-the-race/
Best: precompute best tire segment + DP — O(numLaps^2)
"""

from typing import List


class Solution:
    def minimumFinishTime(self, tires: List[List[int]], changeTime: int, numLaps: int) -> int:
        # without changing, consecutive x laps min time for best tire
        INF = 10**18
        # max useful consecutive laps before time exceeds change + new tire
        minWithoutChange = [INF] * 19
        for f, r in tires:
            t = 0
            cur = f
            for x in range(1, 19):
                t += cur
                if t >= INF:
                    break
                minWithoutChange[x] = min(minWithoutChange[x], t)
                cur *= r
                if cur > changeTime + f:
                    # further consecutive worse than change
                    break
        dp = [INF] * (numLaps + 1)
        dp[0] = -changeTime  # first tire no prior change? adjust: start with changeTime paid virtually
        for i in range(1, numLaps + 1):
            for j in range(1, min(i, 18) + 1):
                if minWithoutChange[j] < INF:
                    dp[i] = min(dp[i], dp[i - j] + changeTime + minWithoutChange[j])
        return dp[numLaps]


def _test():
    s = Solution()
    assert s.minimumFinishTime([[2, 3], [3, 4]], 5, 4) == 21
    assert s.minimumFinishTime([[1, 10], [2, 2], [3, 4]], 6, 5) == 25
    print("LC 2188 OK")


if __name__ == "__main__":
    _test()
