"""
LC 3405. Count the Number of Arrays with K Matching Adjacent Elements — Hard
https://leetcode.com/problems/count-the-number-of-arrays-with-k-matching-adjacent-elements/
Best: Combinatorics — O(n + log MOD)
"""

MOD = 10**9 + 7


class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        # exactly k positions i where arr[i]==arr[i+1]
        # choose k of (n-1) adjacent slots to be equal: C(n-1, k)
        # first element: m choices; for each of n-1-k "changes": m-1 choices
        # for each of k equals: 1 choice
        def modinv(a):
            return pow(a, MOD - 2, MOD)

        def comb(N, K):
            if K < 0 or K > N:
                return 0
            num = den = 1
            for i in range(K):
                num = num * (N - i) % MOD
                den = den * (i + 1) % MOD
            return num * modinv(den) % MOD

        return comb(n - 1, k) * m % MOD * pow(m - 1, n - 1 - k, MOD) % MOD


def _test():
    s = Solution()
    assert s.countGoodArrays(3, 2, 1) == 4
    assert s.countGoodArrays(4, 2, 2) == 6
    assert s.countGoodArrays(5, 2, 0) == 2
    print("LC 3405 OK")


if __name__ == "__main__":
    _test()
