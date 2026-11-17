"""
LC 3336. Find the Number of Subsequences With Equal GCD — Hard
https://leetcode.com/problems/find-the-number-of-subsequences-with-equal-gcd/
Best: 方法一：记忆化搜索
"""

from functools import cache
from math import gcd

class Solution:
    def subsequencePairCount(self, nums: list[int]) -> int:
        @cache
        def dfs(i: int, j: int, k: int) -> int:
            if i < 0:
                return int(j == k)
            return (
                dfs(i - 1, j, k)
                + dfs(i - 1, gcd(nums[i], j), k)
                + dfs(i - 1, j, gcd(nums[i], k))
            ) % mod

        mod = 10**9 + 7
        return (dfs(len(nums) - 1, 0, 0) - 1) % mod

def _test():
    s = Solution()
    assert s.subsequencePairCount(nums=[1, 2, 3, 4]) == 10
    assert s.subsequencePairCount(nums=[10, 20, 30]) == 2
    assert s.subsequencePairCount(nums=[1, 1, 1, 1]) == 50
    print("LC 3336 OK")

if __name__ == "__main__":
    _test()
