"""
LC 1883. Minimum Skips to Arrive at Meeting On Time — Hard
https://leetcode.com/problems/minimum-skips-to-arrive-at-meeting-on-time/
Best: DP min time with skips + ceil rest — O(n^2)
"""


from typing import List
import math


class Solution:
    def minSkips(self, dist: List[int], speed: int, hoursBefore: int) -> int:
        n = len(dist)
        # dp[j] = min time*speed units after considering roads, with j skips
        # work in integers: time in units of 1/speed
        INF = 10**30
        dp = [INF] * (n + 1)
        dp[0] = 0
        for i, d in enumerate(dist):
            ndp = [INF] * (n + 1)
            for j in range(i + 1):
                if dp[j] >= INF:
                    continue
                # skip rest at end of this road (except last)
                if i < n - 1:
                    ndp[j + 1] = min(ndp[j + 1], dp[j] + d)
                    # no skip: ceil to multiple of speed
                    t = dp[j] + d
                    t = (t + speed - 1) // speed * speed
                    ndp[j] = min(ndp[j], t)
                else:
                    ndp[j] = min(ndp[j], dp[j] + d)
                    ndp[j + 1] = min(ndp[j + 1], dp[j] + d)
            dp = ndp
        limit = hoursBefore * speed
        for j in range(n + 1):
            if dp[j] <= limit:
                return j
        return -1


def _test():
    s = Solution()
    assert s.minSkips([1, 3, 2], 4, 2) == 1
    assert s.minSkips([7, 3, 5, 5], 2, 10) == 2
    assert s.minSkips([7, 3, 5, 5], 1, 10) == -1
    print("LC 1883 OK")


if __name__ == "__main__":
    _test()

