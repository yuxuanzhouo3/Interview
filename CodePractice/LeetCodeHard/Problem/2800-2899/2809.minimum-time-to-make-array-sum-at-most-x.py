"""
LC 2809. Minimum Time to Make Array Sum At Most x — Hard
https://leetcode.com/problems/minimum-time-to-make-array-sum-at-most-x/
Best: Sort by nums2 + knapsack DP — O(n^2)
"""

from typing import List


class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        pairs = sorted(zip(nums2, nums1))
        # dp[j] = max total reduction using j operations (best assignment of which j indices to zero at times j..1)
        dp = [0] * (n + 1)
        for i, (b, a) in enumerate(pairs, 1):
            for j in range(i, 0, -1):
                dp[j] = max(dp[j], dp[j - 1] + a + j * b)
        s1, s2 = sum(nums1), sum(nums2)
        for t in range(n + 1):
            if s1 + s2 * t - dp[t] <= x:
                return t
        return -1


def _test():
    s = Solution()
    assert s.minimumTime([1, 2, 3], [1, 2, 3], 4) == 3
    assert s.minimumTime([1, 2, 3], [3, 3, 3], 4) == -1
    print("LC 2809 OK")


if __name__ == "__main__":
    _test()
