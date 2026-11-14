"""
LC 4. Median of Two Sorted Arrays — Hard
https://leetcode.com/problems/median-of-two-sorted-arrays/
Best: Binary partition — O(log(min(m,n)))
"""

from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        A, B = nums1, nums2
        if len(A) > len(B):
            A, B = B, A
        m, n = len(A), len(B)
        lo, hi = 0, m
        half = (m + n + 1) // 2
        while lo <= hi:
            i = (lo + hi) // 2
            j = half - i
            Aleft = A[i - 1] if i else float("-inf")
            Aright = A[i] if i < m else float("inf")
            Bleft = B[j - 1] if j else float("-inf")
            Bright = B[j] if j < n else float("inf")
            if Aleft <= Bright and Bleft <= Aright:
                if (m + n) % 2:
                    return float(max(Aleft, Bleft))
                return (max(Aleft, Bleft) + min(Aright, Bright)) / 2.0
            if Aleft > Bright:
                hi = i - 1
            else:
                lo = i + 1
        raise ValueError("unreachable")


def _test():
    s = Solution()
    assert s.findMedianSortedArrays([1, 3], [2]) == 2.0
    assert s.findMedianSortedArrays([1, 2], [3, 4]) == 2.5
    assert s.findMedianSortedArrays([], [1]) == 1.0
    print("LC 4 OK")


if __name__ == "__main__":
    _test()
