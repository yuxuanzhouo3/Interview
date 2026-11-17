"""
LC 1955. Count Number of Special Subsequences — Hard
https://leetcode.com/problems/count-number-of-special-subsequences/
Best: DP count 0 / 01 / 012 — O(n)
"""

from typing import List

MOD = 10**9 + 7


class Solution:
    def countSpecialSubsequences(self, nums: List[int]) -> int:
        # dp0: subsequences of form 0+
        # dp1: 0+1+
        # dp2: 0+1+2+
        dp0 = dp1 = dp2 = 0
        for x in nums:
            if x == 0:
                dp0 = (dp0 * 2 + 1) % MOD
            elif x == 1:
                dp1 = (dp1 * 2 + dp0) % MOD
            else:  # x == 2
                dp2 = (dp2 * 2 + dp1) % MOD
        return dp2


def _test():
    s = Solution()
    assert s.countSpecialSubsequences([0, 1, 2, 2]) == 3
    assert s.countSpecialSubsequences([2, 2, 0, 0]) == 0
    assert s.countSpecialSubsequences([0, 1, 2, 0, 1, 2]) == 7
    print("LC 1955 OK")


if __name__ == "__main__":
    _test()
