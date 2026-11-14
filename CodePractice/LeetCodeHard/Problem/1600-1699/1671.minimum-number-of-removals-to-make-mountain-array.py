"""
LC 1671. Minimum Number of Removals to Make Mountain Array — Hard
https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/
Best: LIS both sides — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        n = len(nums)

        def lis_len(arr):
            tails = []
            dp = [0] * len(arr)
            for i, x in enumerate(arr):
                j = bisect.bisect_left(tails, x)
                if j == len(tails):
                    tails.append(x)
                else:
                    tails[j] = x
                dp[i] = j + 1
            return dp

        left = lis_len(nums)
        right = lis_len(nums[::-1])[::-1]
        best = 0
        for i in range(n):
            if left[i] >= 2 and right[i] >= 2:
                best = max(best, left[i] + right[i] - 1)
        return n - best


def _test():
    s = Solution()
    assert s.minimumMountainRemovals([1, 3, 1]) == 0
    assert s.minimumMountainRemovals([2, 1, 1, 5, 6, 2, 3, 1]) == 3
    assert s.minimumMountainRemovals([1, 2, 3, 4, 4, 3, 2, 1]) == 1
    print("LC 1671 OK")


if __name__ == "__main__":
    _test()
