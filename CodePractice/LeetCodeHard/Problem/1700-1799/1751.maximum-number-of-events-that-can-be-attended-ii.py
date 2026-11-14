"""
LC 1751. Maximum Number of Events That Can Be Attended II — Hard
https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/
Best: DP + binary search next — O(nk log n)
"""

from typing import List
import bisect


class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:
        events.sort(key=lambda e: e[0])
        n = len(events)
        starts = [e[0] for e in events]
        # dp[i][t]: from event i, attend at most t
        from functools import lru_cache

        @lru_cache(None)
        def dfs(i, t):
            if i == n or t == 0:
                return 0
            # skip
            best = dfs(i + 1, t)
            # take
            j = bisect.bisect_right(starts, events[i][1])
            best = max(best, events[i][2] + dfs(j, t - 1))
            return best

        return dfs(0, k)


def _test():
    s = Solution()
    assert s.maxValue([[1, 2, 4], [3, 4, 3], [2, 3, 1]], 2) == 7
    assert s.maxValue([[1, 2, 4], [3, 4, 3], [2, 3, 10]], 2) == 10
    assert s.maxValue([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], 3) == 9
    print("LC 1751 OK")


if __name__ == "__main__":
    _test()
