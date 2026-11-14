"""
LC 801. Minimum Swaps To Make Sequences Increasing — Hard
Best: DP swap/keep — O(n)
"""

from typing import List


class Solution:
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        keep, swap = 0, 1
        for i in range(1, n):
            nk = ns = float("inf")
            if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:
                nk = keep
                ns = swap + 1
            if nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
                nk = min(nk, swap)
                ns = min(ns, keep + 1)
            keep, swap = nk, ns
        return min(keep, swap)


def _test():
    s = Solution()
    assert s.minSwap([0, 3, 5, 8, 9], [2, 1, 4, 6, 9]) == 1
    print("LC 801 OK")


if __name__ == "__main__":
    _test()
