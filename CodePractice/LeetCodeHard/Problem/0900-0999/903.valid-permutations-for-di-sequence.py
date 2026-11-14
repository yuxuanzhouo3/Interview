"""
LC 903. Valid Permutations for DI Sequence — Hard
https://leetcode.com/problems/valid-permutations-for-di-sequence/
Best: DP with prefix sums — O(n^2)
"""

class Solution:
    def numPermsDISequence(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # dp[j] = ways ending with relative rank j among remaining
        dp = [1] * (n + 1)
        for i, c in enumerate(s):
            ndp = [0] * (n - i)
            if c == "D":
                # sum dp[j+1..] into ndp[j]
                suf = 0
                for j in range(n - i - 1, -1, -1):
                    suf = (suf + dp[j + 1]) % MOD
                    ndp[j] = suf
            else:  # I
                pre = 0
                for j in range(n - i):
                    pre = (pre + dp[j]) % MOD
                    ndp[j] = pre
            dp = ndp
        return sum(dp) % MOD


def _test():
    s = Solution()
    assert s.numPermsDISequence("DID") == 5
    assert s.numPermsDISequence("D") == 1
    assert s.numPermsDISequence("I") == 1
    print("LC 903 OK")


if __name__ == "__main__":
    _test()
