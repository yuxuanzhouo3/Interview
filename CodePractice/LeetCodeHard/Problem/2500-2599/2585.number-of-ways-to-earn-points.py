"""
LC 2585. Number of Ways to Earn Points — Hard
https://leetcode.com/problems/number-of-ways-to-earn-points/
Best: Multiple knapsack DP — O(target * Σ count)
"""

from typing import List


class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        MOD = 10**9 + 7
        dp = [0] * (target + 1)
        dp[0] = 1
        for count, marks in types:
            for t in range(target, -1, -1):
                for c in range(1, count + 1):
                    if t < c * marks:
                        break
                    dp[t] = (dp[t] + dp[t - c * marks]) % MOD
        return dp[target]


def _test():
    s = Solution()
    assert s.waysToReachTarget(6, [[6, 1], [3, 2], [2, 3]]) == 7
    assert s.waysToReachTarget(5, [[50, 1], [50, 2], [50, 5]]) == 4
    assert s.waysToReachTarget(18, [[6, 1], [3, 2], [2, 3]]) == 1
    print("LC 2585 OK")


if __name__ == "__main__":
    _test()
