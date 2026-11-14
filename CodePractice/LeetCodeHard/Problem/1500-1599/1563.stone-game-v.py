"""
LC 1563. Stone Game V — Hard
https://leetcode.com/problems/stone-game-v/
Best: Interval DP + prefix — O(n^3)
"""

from typing import List
from functools import cache


class Solution:
    def stoneGameV(self, stoneValue: List[int]) -> int:
        n = len(stoneValue)
        pref = [0] * (n + 1)
        for i, v in enumerate(stoneValue):
            pref[i + 1] = pref[i] + v

        @cache
        def dp(l, r):
            if l == r:
                return 0
            best = 0
            for m in range(l, r):
                left = pref[m + 1] - pref[l]
                right = pref[r + 1] - pref[m + 1]
                if left < right:
                    best = max(best, left + dp(l, m))
                elif left > right:
                    best = max(best, right + dp(m + 1, r))
                else:
                    best = max(best, left + dp(l, m), right + dp(m + 1, r))
            return best

        return dp(0, n - 1)


def _test():
    s = Solution()
    assert s.stoneGameV([6, 2, 3, 4, 5, 5]) == 18
    assert s.stoneGameV([7, 7, 7, 7, 7, 7, 7]) == 28
    assert s.stoneGameV([4]) == 0
    print("LC 1563 OK")


if __name__ == "__main__":
    _test()
