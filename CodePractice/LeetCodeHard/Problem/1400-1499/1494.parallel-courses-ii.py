"""
LC 1494. Parallel Courses II — Hard
https://leetcode.com/problems/parallel-courses-ii/
Best: Bitmask DP on course subsets — O(3^n)
"""

from typing import List


class Solution:
    def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
        pre = [0] * n
        for a, b in relations:
            pre[b - 1] |= 1 << (a - 1)
        N = 1 << n
        # prereq mask for each subset
        need = [0] * N
        for mask in range(N):
            for i in range(n):
                if mask >> i & 1:
                    need[mask] |= pre[i]
        INF = 10**9
        dp = [INF] * N
        dp[0] = 0
        for mask in range(N):
            if dp[mask] == INF:
                continue
            # courses available: prereqs satisfied and not taken
            can = 0
            for i in range(n):
                if (mask >> i & 1) == 0 and (pre[i] & mask) == pre[i]:
                    can |= 1 << i
            # enumerate subsets of can with size <= k
            sub = can
            while sub:
                if bin(sub).count("1") <= k:
                    dp[mask | sub] = min(dp[mask | sub], dp[mask] + 1)
                sub = (sub - 1) & can
            if can == 0 and mask != N - 1:
                pass
        return dp[N - 1]


def _test():
    s = Solution()
    assert s.minNumberOfSemesters(4, [[2, 1], [3, 1], [1, 4]], 2) == 3
    assert s.minNumberOfSemesters(5, [[2, 1], [3, 1], [4, 1], [1, 5]], 2) == 4
    assert s.minNumberOfSemesters(11, [], 2) == 6
    print("LC 1494 OK")


if __name__ == "__main__":
    _test()
