"""
LC 1799. Maximize Score After N Operations — Hard
https://leetcode.com/problems/maximize-score-after-n-operations/
Best: bitmask DP + GCD — O(2^{2n} n^2)
"""

from typing import List
from math import gcd
from functools import lru_cache


class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)

        @lru_cache(None)
        def dfs(mask):
            bits = bin(mask).count("1")
            op = bits // 2 + 1
            if bits == n:
                return 0
            best = 0
            for i in range(n):
                if mask & (1 << i):
                    continue
                for j in range(i + 1, n):
                    if mask & (1 << j):
                        continue
                    best = max(
                        best,
                        op * gcd(nums[i], nums[j]) + dfs(mask | (1 << i) | (1 << j)),
                    )
            return best

        return dfs(0)


def _test():
    s = Solution()
    assert s.maxScore([1, 2]) == 1
    assert s.maxScore([3, 4, 6, 8]) == 11
    assert s.maxScore([1, 2, 3, 4, 5, 6]) == 14
    print("LC 1799 OK")


if __name__ == "__main__":
    _test()
