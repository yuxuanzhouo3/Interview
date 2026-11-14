"""
LC 2141. Maximum Running Time of N Computers — Hard
https://leetcode.com/problems/maximum-running-time-of-n-computers/
Best: binary search on time — O(m log S)
"""

from typing import List


class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        batteries.sort()
        lo, hi = 0, sum(batteries) // n

        def can(t):
            return sum(min(b, t) for b in batteries) >= n * t

        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


def _test():
    s = Solution()
    assert s.maxRunTime(2, [3, 3, 3]) == 4
    assert s.maxRunTime(2, [1, 1, 1, 1]) == 2
    print("LC 2141 OK")


if __name__ == "__main__":
    _test()
