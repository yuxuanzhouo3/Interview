"""
LC 2514. Count Anagrams — Hard
https://leetcode.com/problems/count-anagrams/
Best: Product of multinomials per word — O(L + Σ n log MOD)
"""

from collections import Counter

MOD = 10**9 + 7


class Solution:
    def countAnagrams(self, s: str) -> int:
        # precompute factorials up to max word len
        words = s.split()
        mx = max(len(w) for w in words)
        fac = [1] * (mx + 1)
        for i in range(1, mx + 1):
            fac[i] = fac[i - 1] * i % MOD
        invfac = [1] * (mx + 1)
        invfac[mx] = pow(fac[mx], MOD - 2, MOD)
        for i in range(mx, 0, -1):
            invfac[i - 1] = invfac[i] * i % MOD

        ans = 1
        for w in words:
            cnt = Counter(w)
            ways = fac[len(w)]
            for c in cnt.values():
                ways = ways * invfac[c] % MOD
            ans = ans * ways % MOD
        return ans


def _test():
    s = Solution()
    assert s.countAnagrams("too hot") == 18
    assert s.countAnagrams("aa") == 1
    print("LC 2514 OK")


if __name__ == "__main__":
    _test()
