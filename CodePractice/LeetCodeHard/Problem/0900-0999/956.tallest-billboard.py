"""
LC 956. Tallest Billboard — Hard
https://leetcode.com/problems/tallest-billboard/
Best: Diff DP knapsack — O(n * sum)
"""

from typing import List


class Solution:
    def tallestBillboard(self, rods: List[int]) -> int:
        # dp[diff] = max common height (taller side contribution tracked via sum)
        # Actually: dp[d] = max height of the shorter? Better classic:
        # dp[diff] = max sum of used rods that achieve (left-right)=diff
        dp = {0: 0}
        for r in rods:
            ndp = dict(dp)
            for d, s in dp.items():
                # put on left
                nd = d + r
                ndp[nd] = max(ndp.get(nd, 0), s + r)
                # put on right
                nd = d - r
                ndp[nd] = max(ndp.get(nd, 0), s + r)
            dp = ndp
        # diff 0: total used / 2 is height
        return dp.get(0, 0) // 2


def _test():
    s = Solution()
    assert s.tallestBillboard([1, 2, 3, 6]) == 6
    assert s.tallestBillboard([1, 2, 3, 4, 5, 6]) == 10
    assert s.tallestBillboard([1, 2]) == 0
    print("LC 956 OK")


if __name__ == "__main__":
    _test()
