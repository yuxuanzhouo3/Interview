"""
LC 3426. Manhattan Distances of All Arrangements of Pieces — Hard
https://leetcode.com/problems/manhattan-distances-of-all-arrangements-of-pieces/
Best: Separable combinatorics on rows/cols — O((m+n) + k log MOD)
"""

MOD = 10**9 + 7


class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        # sum over all ways to choose k cells of sum of pairwise Manhattan distances
        # = C(mn-2, k-2) * sum_{all pairs of cells} manhattan(p,q)
        # manhattan separates into |r1-r2| + |c1-c2|

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

        # sum of |i-j| over all cell pairs:
        # for rows: each pair of distinct rows r1<r2 contributes (r2-r1)*n*n
        # (n choices of col for each row)
        row_sum = 0
        for d in range(1, m):
            row_sum = (row_sum + d * (m - d) % MOD * n % MOD * n) % MOD
        col_sum = 0
        for d in range(1, n):
            col_sum = (col_sum + d * (n - d) % MOD * m % MOD * m) % MOD
        pair_sum = (row_sum + col_sum) % MOD
        return pair_sum * comb(m * n - 2, k - 2) % MOD


def _test():
    s = Solution()
    assert s.distanceSum(2, 2, 2) == 8
    assert s.distanceSum(1, 4, 3) == 20
    print("LC 3426 OK")


if __name__ == "__main__":
    _test()
