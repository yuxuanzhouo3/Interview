"""
LC 2444. Count Subarrays With Fixed Bounds — Hard
https://leetcode.com/problems/count-subarrays-with-fixed-bounds/
Best: Sliding window last positions of minK/maxK/bad — O(n)
"""

from typing import List


class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        ans = 0
        imin = imax = ibad = -1
        for i, x in enumerate(nums):
            if x < minK or x > maxK:
                ibad = i
            if x == minK:
                imin = i
            if x == maxK:
                imax = i
            left = min(imin, imax)
            if left > ibad:
                ans += left - ibad
        return ans


def _test():
    s = Solution()
    assert s.countSubarrays([1, 3, 5, 2, 7, 5], 1, 5) == 2
    assert s.countSubarrays([1, 1, 1, 1], 1, 1) == 10
    print("LC 2444 OK")


if __name__ == "__main__":
    _test()
