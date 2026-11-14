"""
LC 629. K Inverse Pairs Array — Hard
https://leetcode.com/problems/k-inverse-pairs-array/
Best: DP + prefix — O(n*k)
"""

MOD = 10**9 + 7


class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        # dp[j] = ways for current length with j inverse pairs
        dp = [0] * (k + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            ndp = [0] * (k + 1)
            pref = 0
            for j in range(k + 1):
                pref = (pref + dp[j]) % MOD
                if j - i >= 0:
                    pref = (pref - dp[j - i]) % MOD
                ndp[j] = pref
            dp = ndp
        return dp[k]


def _test():
    s = Solution()
    assert s.kInversePairs(3, 0) == 1
    assert s.kInversePairs(3, 1) == 2
    assert s.kInversePairs(2, 2) == 0
    print("LC 629 OK")


if __name__ == "__main__":
    _test()
