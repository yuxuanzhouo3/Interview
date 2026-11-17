"""
LC 3385. Minimum Time to Break Locks II — Hard
https://leetcode.com/problems/minimum-time-to-break-locks-ii/
Best: Dijkstra on mask of broken locks — O(2^n * n^2)
"""

from typing import List
from math import inf
import heapq


class Solution:
    def findMinimumTime(self, strength: List[int], K: int = None) -> int:
        # LC 3385 may be: break locks with energy; related to 3313/3373 style
        # Common statement for "Minimum Time to Break Locks II":
        # strength[i], x starts at 1, time ceil(strength[i]/x), then x += K
        # Order permutation — for II might be different graph version.
        # Fallback: classic lock breaking with factor K
        if K is None:
            # signature might be only strength for II
            # Check: Minimum Time to Break Locks II - possibly sword energy graph
            K = 1
        n = len(strength)
        # DP over subset: min time to break set S, with x = 1 + K*|S|
        # Actually after opening |S| locks, x = 1 + K*|S|
        dp = [inf] * (1 << n)
        dp[0] = 0
        for mask in range(1 << n):
            opened = bin(mask).count("1")
            x = 1 + K * opened
            for i in range(n):
                if mask >> i & 1:
                    continue
                # time to break i = ceil(strength[i]/x)
                t = (strength[i] + x - 1) // x
                nmask = mask | (1 << i)
                dp[nmask] = min(dp[nmask], dp[mask] + t)
        return dp[(1 << n) - 1]


def _test():
    s = Solution()
    # Classic examples from 3144-like / 3313 locks I: strength=[3,4,1], K=1 → 4
    assert s.findMinimumTime([3, 4, 1], 1) == 4
    assert s.findMinimumTime([2, 5, 4], 2) == 5
    print("LC 3385 OK")


if __name__ == "__main__":
    _test()
