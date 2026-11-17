"""
LC 3098. Find the Sum of Subsequence Powers — Hard
https://leetcode.com/problems/find-the-sum-of-subsequence-powers/
Best: Sort + DP on min-diff — O(n^2 k)
"""

from typing import List
from functools import lru_cache


class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        # power = min adjacent diff in sorted subsequence of size k
        # Sum over all such subsequences
        # Enumerate the min-diff value d = nums[j]-nums[i] as the power, count subsequences where this is achieved
        ans = 0
        # DP: for each possible pair as the closest...
        # Standard: dp[i][c][p] too heavy
        # Better: for sorted a, 
        # f(i, left, mini) 
        @lru_cache(None)
        def dp(i, need, prev, bound):
            # sum of powers for choosing `need` more from i..n-1, prev index, current min-diff bound (inf if none)
            if need == 0:
                return bound if bound < 10**18 else 0
            if i == n:
                return 0
            # skip
            res = dp(i + 1, need, prev, bound)
            # take
            if prev == -1:
                res += dp(i + 1, need - 1, i, bound)
            else:
                d = nums[i] - nums[prev]
                nb = d if bound >= 10**18 else min(bound, d)
                res += dp(i + 1, need - 1, i, nb)
            return res % MOD

        # bound as None -> use None; lru needs hashable: use a large sentinel
        INF = 10**18
        return dp(0, k, -1, INF) % MOD


def _test():
    s = Solution()
    assert s.sumOfPowers([1, 2, 3, 4], 3) == 4
    assert s.sumOfPowers([2, 2], 2) == 0
    assert s.sumOfPowers([4, 3, -1], 2) == 10
    print("LC 3098 OK")


if __name__ == "__main__":
    _test()
