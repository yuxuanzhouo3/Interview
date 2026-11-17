"""
LC 1735. Count Ways to Make Array With Product — Hard
https://leetcode.com/problems/count-ways-to-make-array-with-product/
Best: Prime factorization + stars and bars — O(q · √k + π(k)·n)
"""

from typing import List
from math import comb

MOD = 10**9 + 7


class Solution:
    def waysToFillArray(self, queries: List[List[int]]) -> List[int]:
        # precompute factorials for C(n+e-1, e) up to n=10^4, e small (exponents)
        # max exponent for k<=1e4 is small; n+e-1 <= 1e4+14
        N = 10**4 + 20
        fac = [1] * (N + 1)
        for i in range(1, N + 1):
            fac[i] = fac[i - 1] * i % MOD
        invfac = [1] * (N + 1)
        invfac[N] = pow(fac[N], MOD - 2, MOD)
        for i in range(N, 0, -1):
            invfac[i - 1] = invfac[i] * i % MOD

        def C(n, k):
            if k < 0 or k > n:
                return 0
            return fac[n] * invfac[k] % MOD * invfac[n - k] % MOD

        def factorize(x):
            res = []
            d = 2
            while d * d <= x:
                if x % d == 0:
                    c = 0
                    while x % d == 0:
                        x //= d
                        c += 1
                    res.append(c)
                d += 1
            if x > 1:
                res.append(1)
            return res

        ans = []
        for n, k in queries:
            ways = 1
            for e in factorize(k):
                # distribute e indistinguishable primes into n positions: C(e+n-1, e)
                ways = ways * C(e + n - 1, e) % MOD
            ans.append(ways)
        return ans


def _test():
    s = Solution()
    assert s.waysToFillArray([[2, 6], [5, 1], [73, 660]]) == [4, 1, 50734910]
    assert s.waysToFillArray([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]) == [1, 2, 3, 10, 5]
    print("LC 1735 OK")


if __name__ == "__main__":
    _test()
