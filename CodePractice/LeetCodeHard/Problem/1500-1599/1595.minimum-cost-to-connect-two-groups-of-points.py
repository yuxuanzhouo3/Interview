"""
LC 1595. Minimum Cost to Connect Two Groups of Points — Hard
https://leetcode.com/problems/minimum-cost-to-connect-two-groups-of-points/
Best: DP on bitmask of group2 — O(size1 · 2^size2 · size2)
"""

from typing import List


class Solution:
    def connectTwoGroups(self, cost: List[List[int]]) -> int:
        m, n = len(cost), len(cost[0])
        min2 = [min(cost[i][j] for i in range(m)) for j in range(n)]
        INF = 10**18
        dp = [INF] * (1 << n)
        dp[0] = 0
        for i in range(m):
            ndp = [INF] * (1 << n)
            for mask in range(1 << n):
                if dp[mask] == INF:
                    continue
                for j in range(n):
                    nmask = mask | (1 << j)
                    ndp[nmask] = min(ndp[nmask], dp[mask] + cost[i][j])
            dp = ndp
        ans = INF
        for mask in range(1 << n):
            if dp[mask] == INF:
                continue
            extra = 0
            for j in range(n):
                if (mask >> j) & 1 == 0:
                    extra += min2[j]
            ans = min(ans, dp[mask] + extra)
        return ans


def _test():
    s = Solution()
    assert s.connectTwoGroups([[15, 96], [36, 2]]) == 17
    assert s.connectTwoGroups([[1, 3, 5], [4, 1, 1], [1, 5, 3]]) == 4
    assert s.connectTwoGroups([[2, 5, 1], [3, 4, 7], [8, 1, 2], [6, 2, 4], [3, 8, 8]]) == 10
    print("LC 1595 OK")


if __name__ == "__main__":
    _test()
