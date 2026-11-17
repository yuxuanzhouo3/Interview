"""
LC 3231. Minimum Number of Increasing Subsequence to Be Removed — Hard
https://leetcode.com/problems/minimum-number-of-increasing-subsequence-to-be-removed/
Best: Longest non-increasing subsequence length (patience) — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Min increasing subsequences to cover = length of longest decreasing subsequence.
        # Equivalently: LIS length on negated array for non-increasing? 
        # Longest strictly decreasing: patience on -nums with strict.
        # For covering by increasing subsequences: LDS (non-increasing or strict decreasing).
        # LeetCode 3231: remove strictly increasing subsequences; answer = length of longest non-increasing subsequence.
        tails = []
        for x in nums:
            # longest non-increasing: find first tail < x (strict) to replace — use lower_bound on reversed order
            # maintain tails as non-increasing sequence candidates in "patience" for LDS non-increasing:
            # bisect_right on negated
            i = bisect.bisect_right(tails, -x)
            if i == len(tails):
                tails.append(-x)
            else:
                tails[i] = -x
        return len(tails)


def _test():
    s = Solution()
    assert s.minOperations([1, 2, 3, 4]) == 1
    assert s.minOperations([4, 3, 2, 1]) == 4
    assert s.minOperations([1, 3, 2, 4]) == 2
    print("LC 3231 OK")


if __name__ == "__main__":
    _test()
