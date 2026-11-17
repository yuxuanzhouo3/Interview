"""
LC 3269. Constructing Two Increasing Arrays — Hard
https://leetcode.com/problems/constructing-two-increasing-arrays/
Best: 2D DP min largest — O(mn)
"""

from math import inf
from typing import List


class Solution:
    def minLargest(self, nums1: List[int], nums2: List[int]) -> int:
        m, n = len(nums1), len(nums2)
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0

        def nxt(prev: int, num: int) -> int:
            return prev + (2 if prev % 2 == num else 1)

        for i in range(m + 1):
            for j in range(n + 1):
                if i > 0:
                    dp[i][j] = min(dp[i][j], nxt(dp[i - 1][j], nums1[i - 1]))
                if j > 0:
                    dp[i][j] = min(dp[i][j], nxt(dp[i][j - 1], nums2[j - 1]))
        return dp[m][n]


def _test():
    s = Solution()
    assert s.minLargest([], [1, 0, 1, 1]) == 5
    assert s.minLargest([0, 1, 0, 1], [1, 0, 0, 1]) == 9
    assert s.minLargest([0, 1, 0, 0, 1], [0, 0, 0, 1]) == 13
    print("LC 3269 OK")


if __name__ == "__main__":
    _test()
