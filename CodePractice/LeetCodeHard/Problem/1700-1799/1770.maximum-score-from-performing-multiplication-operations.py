"""
LC 1770. Maximum Score from Performing Multiplication Operations — Hard
https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations/
Best: DP dp[i][l] — O(m^2)
"""

from typing import List


class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        n, m = len(nums), len(multipliers)
        # dp[i][l]: max score using first i multipliers with left l taken
        NEG = -(10**18)
        dp = [[NEG] * (m + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(m):
            mul = multipliers[i]
            for l in range(i + 1):
                cur = dp[i][l]
                if cur == NEG:
                    continue
                # take left nums[l]
                dp[i + 1][l + 1] = max(dp[i + 1][l + 1], cur + nums[l] * mul)
                # take right: right index = n - 1 - (i - l)
                r = n - 1 - (i - l)
                dp[i + 1][l] = max(dp[i + 1][l], cur + nums[r] * mul)
        return max(dp[m])


def _test():
    s = Solution()
    assert s.maximumScore([1, 2, 3], [3, 2, 1]) == 14
    assert s.maximumScore([-5, -3, -3, -2, 7, 1], [-10, -5, 3, 4, 6]) == 102
    print("LC 1770 OK")


if __name__ == "__main__":
    _test()
