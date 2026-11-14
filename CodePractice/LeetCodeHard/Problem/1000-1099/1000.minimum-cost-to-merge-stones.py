"""
LC 1000. Minimum Cost to Merge Stones — Hard
https://leetcode.com/problems/minimum-cost-to-merge-stones/
Best: Interval DP — O(n^3 / K)
"""

from typing import List


class Solution:
    def mergeStones(self, stones: List[int], K: int) -> int:
        n = len(stones)
        if (n - 1) % (K - 1) != 0:
            return -1
        pref = [0] * (n + 1)
        for i, x in enumerate(stones):
            pref[i + 1] = pref[i] + x
        INF = 10**18
        dp = [[0] * n for _ in range(n)]
        for length in range(K, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = INF
                for m in range(i, j, K - 1):
                    dp[i][j] = min(dp[i][j], dp[i][m] + dp[m + 1][j])
                if (j - i) % (K - 1) == 0:
                    dp[i][j] += pref[j + 1] - pref[i]
        return dp[0][n - 1]


def _test():
    s = Solution()
    assert s.mergeStones([3, 2, 4, 1], 2) == 20
    assert s.mergeStones([3, 2, 4, 1], 3) == -1
    assert s.mergeStones([3, 5, 1, 2, 6], 3) == 25
    print("LC 1000 OK")


if __name__ == "__main__":
    _test()
