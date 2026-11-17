"""
LC 1478. Allocate Mailboxes — Hard
https://leetcode.com/problems/allocate-mailboxes/
Best: Median cost + DP — O(n^2 · k)
"""

from typing import List


class Solution:
    def minDistance(self, houses: List[int], k: int) -> int:
        houses.sort()
        n = len(houses)
        # cost[i][j] = min cost put 1 mailbox for houses[i..j]
        cost = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                mid = houses[(i + j) // 2]
                cost[i][j] = sum(abs(houses[t] - mid) for t in range(i, j + 1))
        INF = 10**18
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            for t in range(1, min(k, i) + 1):
                for p in range(t - 1, i):
                    dp[i][t] = min(dp[i][t], dp[p][t - 1] + cost[p][i - 1])
        return dp[n][k]


def _test():
    s = Solution()
    assert s.minDistance([1, 4, 8, 10, 20], 3) == 5
    assert s.minDistance([2, 3, 5, 12, 18], 2) == 9
    print("LC 1478 OK")


if __name__ == "__main__":
    _test()
