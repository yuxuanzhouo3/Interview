"""
LC 1449. Form Largest Integer With Digits That Add up to Target — Hard
https://leetcode.com/problems/form-largest-integer-with-digits-that-add-up-to-target/
Best: DP knapsack on cost, then lexicographic reconstruct — O(target · 9)
"""

from typing import List


class Solution:
    def largestNumber(self, cost: List[int], target: int) -> str:
        # dp[t] = best string with cost sum t
        dp = [None] * (target + 1)
        dp[0] = ""
        for t in range(1, target + 1):
            best = None
            for d in range(9, 0, -1):
                c = cost[d - 1]
                if t >= c and dp[t - c] is not None:
                    cand = str(d) + dp[t - c]
                    if best is None or len(cand) > len(best) or (len(cand) == len(best) and cand > best):
                        best = cand
            dp[t] = best
        return dp[target] if dp[target] is not None else "0"


def _test():
    s = Solution()
    assert s.largestNumber([4, 3, 2, 5, 6, 7, 2, 5, 5], 9) == "7772"
    assert s.largestNumber([7, 6, 5, 5, 5, 6, 8, 7, 8], 12) == "85"
    assert s.largestNumber([2, 4, 6, 2, 4, 6, 4, 4, 4], 5) == "0"
    print("LC 1449 OK")


if __name__ == "__main__":
    _test()
