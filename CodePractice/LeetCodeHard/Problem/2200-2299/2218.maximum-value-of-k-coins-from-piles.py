"""
LC 2218. Maximum Value of K Coins From Piles — Hard
https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/
Best: Group knapsack DP with prefix sums — O(k * Σpile)
"""

from typing import List


class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        dp = [0] + [-10**18] * k
        for pile in piles:
            pref = [0]
            for x in pile:
                pref.append(pref[-1] + x)
            ndp = dp[:]
            m = len(pile)
            for used in range(k + 1):
                if dp[used] < -10**17:
                    continue
                for take in range(1, min(m, k - used) + 1):
                    ndp[used + take] = max(ndp[used + take], dp[used] + pref[take])
            dp = ndp
        return dp[k]


def _test():
    s = Solution()
    assert s.maxValueOfCoins([[1, 100, 3], [7, 8, 9]], 2) == 101
    assert s.maxValueOfCoins([[100], [100], [100], [100], [100], [100], [1, 1, 1, 1, 1, 1, 700]], 7) == 706
    print("LC 2218 OK")


if __name__ == "__main__":
    _test()
