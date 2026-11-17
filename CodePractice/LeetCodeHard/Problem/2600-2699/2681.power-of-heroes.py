"""
LC 2681. Power of Heroes — Hard
https://leetcode.com/problems/power-of-heroes/
Best: Sort + prefix of mins contribution — O(n log n)
"""

from typing import List

MOD = 10**9 + 7


class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        ans = 0
        pref = 0  # sum of mins for subsets ending with previous as ... 
        # For each nums[i] as maximum: power = max^2 * min over nonempty subsets of left+self
        # Equivalent: ans += nums[i]^2 * (nums[i] + sum_{j<i} nums[j] * 2^{i-j-1})
        for i, x in enumerate(nums):
            ans = (ans + x * x % MOD * (x + pref) % MOD) % MOD
            pref = (pref * 2 + x) % MOD
        return ans


def _test():
    s = Solution()
    assert s.sumOfPower([2, 1, 4]) == 141
    assert s.sumOfPower([1, 1, 1]) == 7
    print("LC 2681 OK")


if __name__ == "__main__":
    _test()
