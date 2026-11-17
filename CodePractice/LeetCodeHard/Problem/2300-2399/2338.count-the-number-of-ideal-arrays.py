"""
LC 2338. Count the Number of Ideal Arrays — Hard
https://leetcode.com/problems/count-the-number-of-ideal-arrays/
Best: Prime factorize + stars-and-bars C(n+e-1,e) — O(maxValue log + n)
"""


class Solution:
    def idealArrays(self, n: int, maxValue: int) -> int:
        MOD = 10**9 + 7
        # precompute factorials for C(n+c-1, c) where c is exponent count
        max_exp = 14  # 2^14 > 10^4
        N = n + max_exp
        fac = [1] * (N + 1)
        for i in range(1, N + 1):
            fac[i] = fac[i - 1] * i % MOD
        invfac = [1] * (N + 1)
        invfac[N] = pow(fac[N], MOD - 2, MOD)
        for i in range(N, 0, -1):
            invfac[i - 1] = invfac[i] * i % MOD

        def C(a, b):
            if b < 0 or b > a:
                return 0
            return fac[a] * invfac[b] % MOD * invfac[a - b] % MOD

        # sieve smallest prime factor
        spf = list(range(maxValue + 1))
        for i in range(2, int(maxValue**0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, maxValue + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        ans = 0
        for v in range(1, maxValue + 1):
            x = v
            ways = 1
            while x > 1:
                p = spf[x]
                e = 0
                while x % p == 0:
                    x //= p
                    e += 1
                # distribute e indistinguishable increments into n-1 gaps => C(n-1+e, e)
                ways = ways * C(n - 1 + e, e) % MOD
            ans = (ans + ways) % MOD
        return ans


def _test():
    s = Solution()
    assert s.idealArrays(2, 5) == 10
    assert s.idealArrays(5, 3) == 11
    print("LC 2338 OK")


if __name__ == "__main__":
    _test()

