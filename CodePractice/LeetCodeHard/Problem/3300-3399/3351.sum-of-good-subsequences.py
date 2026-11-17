"""
LC 3351. Sum of Good Subsequences — Hard
https://leetcode.com/problems/sum-of-good-subsequences/
Best: DP by value with neighbor counts — O(n + U)
"""

from typing import List
from collections import defaultdict


class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        # good: adjacent elems differ by 1
        # cnt[v] = number of good subsequences ending with v
        # ssum[v] = sum of elements over those subsequences (contribution of all positions)
        cnt = defaultdict(int)
        ssum = defaultdict(int)
        ans = 0
        for x in nums:
            c = 1
            sm = x
            for y in (x - 1, x + 1):
                c = (c + cnt[y]) % MOD
                sm = (sm + ssum[y] + cnt[y] * x) % MOD
            cnt[x] = (cnt[x] + c) % MOD
            ssum[x] = (ssum[x] + sm) % MOD
            ans = (ans + sm) % MOD
        return ans


def _test():
    s = Solution()
    assert s.sumOfGoodSubsequences([1, 2, 1]) == 14
    assert s.sumOfGoodSubsequences([3, 4, 5]) == 40
    print("LC 3351 OK")


if __name__ == "__main__":
    _test()
