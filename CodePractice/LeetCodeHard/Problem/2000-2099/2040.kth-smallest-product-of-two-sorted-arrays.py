"""
LC 2040. Kth Smallest Product of Two Sorted Arrays — Hard
https://leetcode.com/problems/kth-smallest-product-of-two-sorted-arrays/
Best: binary search product + count with two pointers — O((n+m) log R)
"""

from typing import List


class Solution:
    def kthSmallestProduct(self, nums1: List[int], nums2: List[int], k: int) -> int:
        def count_leq(x: int) -> int:
            cnt = 0
            for a in nums1:
                if a > 0:
                    # nums2[j] <= x // a
                    lo, hi = 0, len(nums2)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if a * nums2[mid] <= x:
                            lo = mid + 1
                        else:
                            hi = mid
                    cnt += lo
                elif a < 0:
                    # products decrease as nums2 increases; find first j with a*nums2[j] <= x
                    lo, hi = 0, len(nums2)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if a * nums2[mid] <= x:
                            hi = mid
                        else:
                            lo = mid + 1
                    cnt += len(nums2) - lo
                else:
                    if x >= 0:
                        cnt += len(nums2)
            return cnt

        lo, hi = -(10**10), 10**10
        while lo < hi:
            mid = (lo + hi) // 2
            if count_leq(mid) < k:
                lo = mid + 1
            else:
                hi = mid
        return lo


def _test():
    s = Solution()
    assert s.kthSmallestProduct([2, 5], [3, 4], 2) == 8
    assert s.kthSmallestProduct([-4, -2, 0, 3], [2, 4], 6) == 0
    assert s.kthSmallestProduct([-2, -1, 0, 1, 2], [-3, -1, 2, 4, 5], 3) == -6
    print("LC 2040 OK")


if __name__ == "__main__":
    _test()
