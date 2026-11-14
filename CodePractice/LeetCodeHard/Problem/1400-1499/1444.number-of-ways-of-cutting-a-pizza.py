"""
LC 1444. Number of Ways of Cutting a Pizza — Hard
https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/
Best: Prefix apples + memo DP — O(k m n (m+n))
"""

from typing import List
from functools import cache


class Solution:
    def ways(self, pizza: List[str], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(pizza), len(pizza[0])
        apples = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                apples[i][j] = (
                    apples[i + 1][j]
                    + apples[i][j + 1]
                    - apples[i + 1][j + 1]
                    + (pizza[i][j] == "A")
                )

        @cache
        def dp(r, c, cuts):
            if apples[r][c] == 0:
                return 0
            if cuts == 0:
                return 1
            ans = 0
            for nr in range(r + 1, m):
                if apples[r][c] - apples[nr][c] > 0:
                    ans = (ans + dp(nr, c, cuts - 1)) % MOD
            for nc in range(c + 1, n):
                if apples[r][c] - apples[r][nc] > 0:
                    ans = (ans + dp(r, nc, cuts - 1)) % MOD
            return ans

        return dp(0, 0, k - 1)


def _test():
    s = Solution()
    assert s.ways(["A..", "AAA", "..."], 3) == 3
    assert s.ways(["A..", "AA.", "..."], 3) == 1
    assert s.ways(["A..", "A..", "..."], 1) == 1
    print("LC 1444 OK")


if __name__ == "__main__":
    _test()
