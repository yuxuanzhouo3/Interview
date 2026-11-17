"""
LC 2318. Number of Distinct Roll Sequences — Hard
https://leetcode.com/problems/number-of-distinct-roll-sequences/
Best: DP gcd consecutive dice faces — O(n*6*6*6)
"""


from math import gcd


class Solution:
    def distinctSequences(self, n: int) -> int:
        MOD = 10**9 + 7
        if n == 1:
            return 6
        # dp[prev2][prev1] after placing
        dp = [[0] * 7 for _ in range(7)]
        for a in range(1, 7):
            for b in range(1, 7):
                if a != b and gcd(a, b) == 1:
                    dp[a][b] = 1
        for _ in range(3, n + 1):
            ndp = [[0] * 7 for _ in range(7)]
            for a in range(1, 7):
                for b in range(1, 7):
                    if dp[a][b] == 0:
                        continue
                    for c in range(1, 7):
                        if c != b and c != a and gcd(b, c) == 1:
                            ndp[b][c] = (ndp[b][c] + dp[a][b]) % MOD
            dp = ndp
        return sum(sum(row) for row in dp) % MOD


def _test():
    s = Solution()
    assert s.distinctSequences(1) == 6
    assert s.distinctSequences(2) == 22
    assert s.distinctSequences(4) == 184
    print("LC 2318 OK")


if __name__ == "__main__":
    _test()

