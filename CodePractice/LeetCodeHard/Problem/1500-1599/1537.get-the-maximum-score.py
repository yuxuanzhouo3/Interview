"""
LC 1537. Get the Maximum Score — Hard
https://leetcode.com/problems/get-the-maximum-score/
Best: Two pointers on sorted unique paths — O(n+m)
"""

from typing import List


class Solution:
    def maxSum(self, nums1: List[int], nums2: List[int]) -> int:
        MOD = 10**9 + 7
        i = j = 0
        s1 = s2 = 0
        n, m = len(nums1), len(nums2)
        while i < n or j < m:
            if i < n and (j == m or nums1[i] < nums2[j]):
                s1 += nums1[i]
                i += 1
            elif j < m and (i == n or nums2[j] < nums1[i]):
                s2 += nums2[j]
                j += 1
            else:
                s1 = s2 = max(s1, s2) + nums1[i]
                i += 1
                j += 1
        return max(s1, s2) % MOD


def _test():
    s = Solution()
    assert s.maxSum([2, 4, 5, 8, 10], [4, 6, 8, 9]) == 30
    assert s.maxSum([1, 3, 5, 7, 9], [3, 5, 100]) == 109
    assert s.maxSum([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == 40
    print("LC 1537 OK")


if __name__ == "__main__":
    _test()
