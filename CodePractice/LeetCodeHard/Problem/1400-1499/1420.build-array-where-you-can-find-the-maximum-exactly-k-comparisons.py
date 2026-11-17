"""
LC 1420. Build Array Where You Can Find The Maximum Exactly K Comparisons — Hard
https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/
Best: DP — O(n·m·k)
"""


class Solution:
    def numOfArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        # dp[i][c][mx]: length i, cost c, max mx
        # compress i
        dp = [[0] * (m + 1) for _ in range(k + 1)]
        for mx in range(1, m + 1):
            dp[1][mx] = 1
        for _ in range(2, n + 1):
            ndp = [[0] * (m + 1) for _ in range(k + 1)]
            # prefix sums for transitions when new max
            for c in range(1, k + 1):
                pref = 0
                for mx in range(1, m + 1):
                    # append <= mx: mx choices of value 1..mx keeping max
                    ndp[c][mx] = (ndp[c][mx] + dp[c][mx] * mx) % MOD
                    # append new max = mx: sum dp[c-1][1..mx-1]
                    ndp[c][mx] = (ndp[c][mx] + pref) % MOD
                    pref = (pref + dp[c - 1][mx]) % MOD
            dp = ndp
        return sum(dp[k][mx] for mx in range(1, m + 1)) % MOD


def _test():
    s = Solution()
    assert s.numOfArrays(2, 3, 1) == 6
    assert s.numOfArrays(5, 2, 3) == 0
    assert s.numOfArrays(9, 1, 1) == 1
    print("LC 1420 OK")


if __name__ == "__main__":
    _test()
