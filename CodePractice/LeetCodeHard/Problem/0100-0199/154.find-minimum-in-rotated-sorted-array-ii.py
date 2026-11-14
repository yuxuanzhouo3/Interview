"""
LC 154. Find Minimum in Rotated Sorted Array II — Hard
Best: Binary search with duplicates — O(n) worst, O(log n) typical
"""

from typing import List


class Solution:
    def findMin(self, nums: List[int]) -> int:
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] > nums[hi]:
                lo = mid + 1
            elif nums[mid] < nums[hi]:
                hi = mid
            else:
                hi -= 1
        return nums[lo]


def _test():
    s = Solution()
    assert s.findMin([1, 3, 5]) == 1
    assert s.findMin([2, 2, 2, 0, 1]) == 0
    print("LC 154 OK")


if __name__ == "__main__":
    _test()
