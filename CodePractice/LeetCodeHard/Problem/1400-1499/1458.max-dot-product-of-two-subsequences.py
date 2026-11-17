"""
LC 1458. Max Dot Product of Two Subsequences — Hard
https://leetcode.com/problems/max-dot-product-of-two-subsequences/
Best: DP like LCS — O(nm)
"""

from typing import List


class Solution:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        n, m = len(nums1), len(nums2)
        INF = 10**18
        dp = [[-INF] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                prod = nums1[i - 1] * nums2[j - 1]
                dp[i][j] = max(
                    prod,
                    dp[i - 1][j - 1] + prod,
                    dp[i - 1][j],
                    dp[i][j - 1],
                )
        return dp[n][m]


def _test():
    s = Solution()
    assert s.maxDotProduct([2, 1, -2, 5], [3, 0, -6]) == 18
    assert s.maxDotProduct([3, -2], [2, -6, 7]) == 21
    assert s.maxDotProduct([-1, -1], [1, 1]) == -1
    print("LC 1458 OK")


if __name__ == "__main__":
    _test()
