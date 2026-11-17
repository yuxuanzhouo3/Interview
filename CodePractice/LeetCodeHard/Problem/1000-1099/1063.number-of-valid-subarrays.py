"""
LC 1063. Number of Valid Subarrays — Hard
https://leetcode.com/problems/number-of-valid-subarrays/
Tags: Array, Monotonic Stack
Best: 单调栈：以左端为最小值的子数组个数
"""

from typing import List


class Solution:
    def validSubarrays(self, nums: List[int]) -> int:
        # subarrays where leftmost is the strict minimum (or non-strict: leftmost <= all)
        # LC1063: A subarray is valid if for every i < j in subarray, nums[i] <= nums[j] wait
        # Actually: valid if the leftmost element is <= every other element in the subarray.
        n = len(nums)
        stack = []
        ans = 0
        for i, x in enumerate(nums):
            while stack and nums[stack[-1]] > x:
                j = stack.pop()
                ans += i - j
            stack.append(i)
        while stack:
            j = stack.pop()
            ans += n - j
        return ans


def _test():
    sol = Solution()
    assert sol.validSubarrays([1, 4, 2, 5, 3]) == 11
    assert sol.validSubarrays([3, 2, 1]) == 3
    assert sol.validSubarrays([2, 2, 2]) == 6
    print("LC 1063 OK")


if __name__ == "__main__":
    _test()

