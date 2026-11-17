"""
LC 1259. Handshakes That Don't Cross — Hard
https://leetcode.com/problems/handshakes-that-dont-cross/
Tags: Math, DP, Catalan
Best: 卡特兰数 DP
"""

class Solution:
    def numberOfWays(self, numPeople: int) -> int:
        MOD = 10**9 + 7
        n = numPeople // 2
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(i):
                dp[i] = (dp[i] + dp[j] * dp[i - 1 - j]) % MOD
        return dp[n]


def _test():
    sol = Solution()
    assert sol.numberOfWays(2) == 1
    assert sol.numberOfWays(4) == 2
    assert sol.numberOfWays(6) == 5
    print("LC 1259 OK")


if __name__ == "__main__":
    _test()

