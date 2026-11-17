"""
LC 1330. Reverse Subarray To Maximize Array Value — Hard
https://leetcode.com/problems/reverse-subarray-to-maximize-array-value/
Best: Math on adjacent diffs — O(n)
"""

from typing import List


class Solution:
    def maxValueAfterReverse(self, nums: List[int]) -> int:
        n = len(nums)
        total = sum(abs(nums[i] - nums[i + 1]) for i in range(n - 1))
        # reverse [0..i]
        best = 0
        for i in range(1, n):
            best = max(best, abs(nums[0] - nums[i]) - abs(nums[i] - nums[i - 1]))
        # reverse [i..n-1]
        for i in range(n - 1):
            best = max(best, abs(nums[n - 1] - nums[i]) - abs(nums[i + 1] - nums[i]))
        # reverse middle: maximize |a-d|+|b-c| - |a-b|-|c-d| via min/max pairing
        max1 = float("-inf")
        min1 = float("inf")
        for i in range(n - 1):
            a, b = nums[i], nums[i + 1]
            max1 = max(max1, min(a, b))
            min1 = min(min1, max(a, b))
        best = max(best, 2 * (max1 - min1))
        return total + best


def _test():
    s = Solution()
    assert s.maxValueAfterReverse([2, 3, 1, 5, 4]) == 10
    assert s.maxValueAfterReverse([2, 4, 9, 24, 2, 1, 10]) == 68
    print("LC 1330 OK")


if __name__ == "__main__":
    _test()
