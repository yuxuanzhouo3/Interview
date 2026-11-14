"""
LC 1269. Number of Ways to Stay in the Same Place After Some Steps — Hard
https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/
Best: DP on positions — O(steps * min(arrLen, steps))
"""


class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        MOD = 10**9 + 7
        max_pos = min(arrLen, steps // 2 + 1)
        dp = [0] * max_pos
        dp[0] = 1
        for _ in range(steps):
            ndp = [0] * max_pos
            for i in range(max_pos):
                ndp[i] = dp[i]
                if i > 0:
                    ndp[i] = (ndp[i] + dp[i - 1]) % MOD
                if i + 1 < max_pos:
                    ndp[i] = (ndp[i] + dp[i + 1]) % MOD
            dp = ndp
        return dp[0]


def _test():
    s = Solution()
    assert s.numWays(3, 2) == 4
    assert s.numWays(2, 4) == 2
    assert s.numWays(4, 2) == 8
    print("LC 1269 OK")


if __name__ == "__main__":
    _test()
