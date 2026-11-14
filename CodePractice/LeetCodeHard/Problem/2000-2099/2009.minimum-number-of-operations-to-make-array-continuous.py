"""
LC 2009. Minimum Number of Operations to Make Array Continuous — Hard
https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/
Best: unique sort + sliding window — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        a = sorted(set(nums))
        ans = n
        for i, x in enumerate(a):
            # window [x, x+n-1]
            j = bisect.bisect_right(a, x + n - 1)
            ans = min(ans, n - (j - i))
        return ans


def _test():
    s = Solution()
    assert s.minOperations([4, 2, 5, 3]) == 0
    assert s.minOperations([1, 2, 3, 5, 6]) == 1
    assert s.minOperations([1, 10, 100, 1000]) == 3
    print("LC 2009 OK")


if __name__ == "__main__":
    _test()
