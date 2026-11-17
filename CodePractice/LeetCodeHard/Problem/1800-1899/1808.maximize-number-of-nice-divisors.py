"""
LC 1808. Maximize Number of Nice Divisors — Hard
https://leetcode.com/problems/maximize-number-of-nice-divisors/
Best: break into 3s (integer break) + modpow — O(log n)
"""

MOD = 10**9 + 7


class Solution:
    def maxNiceDivisors(self, primeFactors: int) -> int:
        if primeFactors <= 3:
            return primeFactors
        # maximize product of parts summing to primeFactors; prefer 3s
        q, r = divmod(primeFactors, 3)
        if r == 0:
            return pow(3, q, MOD)
        if r == 1:
            # 3+1 -> 2+2
            return pow(3, q - 1, MOD) * 4 % MOD
        # r == 2
        return pow(3, q, MOD) * 2 % MOD


def _test():
    s = Solution()
    assert s.maxNiceDivisors(5) == 6
    assert s.maxNiceDivisors(8) == 18
    print("LC 1808 OK")


if __name__ == "__main__":
    _test()
