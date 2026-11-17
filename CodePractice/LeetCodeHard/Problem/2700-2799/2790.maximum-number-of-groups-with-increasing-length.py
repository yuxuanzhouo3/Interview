"""
LC 2790. Maximum Number of Groups With Increasing Length — Hard
https://leetcode.com/problems/maximum-number-of-groups-with-increasing-length/
Best: Sort + greedy / binary — O(n log n)
"""

from typing import List


class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        ans = 0
        total = 0
        for x in usageLimits:
            total += x
            if total >= (ans + 1) * (ans + 2) // 2:
                ans += 1
        return ans


def _test():
    s = Solution()
    assert s.maxIncreasingGroups([1, 2, 5]) == 3
    assert s.maxIncreasingGroups([2, 1, 2]) == 2
    assert s.maxIncreasingGroups([1, 1]) == 1
    print("LC 2790 OK")


if __name__ == "__main__":
    _test()

