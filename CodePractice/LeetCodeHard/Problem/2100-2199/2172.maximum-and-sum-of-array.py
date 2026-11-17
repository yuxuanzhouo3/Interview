"""
LC 2172. Maximum AND Sum of Array — Hard
https://leetcode.com/problems/maximum-and-sum-of-array/
Best: bitmask DP on 2*numSlots positions — O(n * 2^(2*numSlots))
"""

from typing import List
from functools import lru_cache


class Solution:
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        n = len(nums)
        m = 2 * numSlots

        @lru_cache(None)
        def dp(mask: int) -> int:
            i = bin(mask).count("1")
            if i == n:
                return 0
            best = 0
            for pos in range(m):
                if mask & (1 << pos) == 0:
                    slot = pos // 2 + 1
                    best = max(best, (nums[i] & slot) + dp(mask | (1 << pos)))
            return best

        return dp(0)


def _test():
    s = Solution()
    assert s.maximumANDSum([1, 2, 3, 4, 5, 6], 3) == 9
    assert s.maximumANDSum([1, 3, 10, 4, 7, 1], 9) == 24
    print("LC 2172 OK")


if __name__ == "__main__":
    _test()
