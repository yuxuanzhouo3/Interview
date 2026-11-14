"""
LC 1547. Minimum Cost to Cut a Stick — Hard
https://leetcode.com/problems/minimum-cost-to-cut-a-stick/
Best: Interval DP — O(m^3)
"""

from typing import List
from functools import cache


class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        arr = [0] + sorted(cuts) + [n]
        m = len(arr)

        @cache
        def dp(i, j):
            if j - i <= 1:
                return 0
            best = min(dp(i, k) + dp(k, j) for k in range(i + 1, j))
            return best + arr[j] - arr[i]

        return dp(0, m - 1)


def _test():
    s = Solution()
    assert s.minCost(7, [1, 3, 4, 5]) == 16
    assert s.minCost(9, [5, 6, 1, 4, 2]) == 22
    print("LC 1547 OK")


if __name__ == "__main__":
    _test()
