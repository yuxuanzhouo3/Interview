"""
LC 1434. Number of Ways to Wear Different Hats to Each Other — Hard
https://leetcode.com/problems/number-of-ways-to-wear-different-hats-to-each-other/
Best: DP on people mask by hats — O(40 · 2^n · n)
"""

from typing import List


class Solution:
    def numberWays(self, hats: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(hats)
        hat_to_people = [[] for _ in range(41)]
        for p, hs in enumerate(hats):
            for h in hs:
                hat_to_people[h].append(p)
        # dp[mask] = ways to assign processed hats to people in mask
        dp = [0] * (1 << n)
        dp[0] = 1
        for h in range(1, 41):
            ndp = dp[:]
            for mask in range(1 << n):
                if dp[mask] == 0:
                    continue
                for p in hat_to_people[h]:
                    if (mask >> p) & 1 == 0:
                        ndp[mask | (1 << p)] = (ndp[mask | (1 << p)] + dp[mask]) % MOD
            dp = ndp
        return dp[(1 << n) - 1]


def _test():
    s = Solution()
    assert s.numberWays([[3, 4], [4, 5], [5]]) == 1
    assert s.numberWays([[3, 5, 1], [3, 5]]) == 4
    assert s.numberWays([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]) == 24
    print("LC 1434 OK")


if __name__ == "__main__":
    _test()
