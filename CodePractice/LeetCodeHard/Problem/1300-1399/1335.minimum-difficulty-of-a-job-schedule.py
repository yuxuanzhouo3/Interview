"""
LC 1335. Minimum Difficulty of a Job Schedule — Hard
https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/
Best: DP days × jobs — O(n^2 d)
"""

from typing import List


class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        n = len(jobDifficulty)
        if n < d:
            return -1
        INF = 10**9
        dp = [INF] * (n + 1)
        dp[0] = 0
        for day in range(1, d + 1):
            ndp = [INF] * (n + 1)
            for i in range(day, n + 1):
                mx = 0
                for j in range(i - 1, day - 2, -1):
                    mx = max(mx, jobDifficulty[j])
                    ndp[i] = min(ndp[i], dp[j] + mx)
            dp = ndp
        return dp[n]


def _test():
    s = Solution()
    assert s.minDifficulty([6, 5, 4, 3, 2, 1], 2) == 7
    assert s.minDifficulty([9, 9, 9], 4) == -1
    assert s.minDifficulty([1, 1, 1], 3) == 3
    print("LC 1335 OK")


if __name__ == "__main__":
    _test()
