"""
LC 1467. Probability of a Two Boxes Having the Same Number of Distinct Balls — Hard
https://leetcode.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/
Best: DFS enumerate distributions — O(∏(balls_i+1))
"""

from math import comb
from typing import List


class Solution:
    def getProbability(self, balls: List[int]) -> float:
        n = sum(balls) // 2
        good = 0

        def dfs(i: int, left_cnt: int, left_colors: int, right_colors: int, ways: int):
            nonlocal good
            if left_cnt > n:
                return
            if i == len(balls):
                if left_cnt == n and left_colors == right_colors:
                    good += ways
                return
            b = balls[i]
            for take in range(b + 1):
                if left_cnt + take > n:
                    break
                lc = left_colors + (1 if take > 0 else 0)
                rc = right_colors + (1 if take < b else 0)
                w = ways * comb(b, take)
                dfs(i + 1, left_cnt + take, lc, rc, w)

        dfs(0, 0, 0, 0, 1)
        # total ways to split into two boxes of n each:
        # each color's balls independently assigned; total = C(2n, n) * product? 
        # Actually: total ways = product over colors C(balls_i, k) summed = C(2n, n)
        # But we counted ways as product of C(b, take); sum = C(2n, n)
        return good / comb(2 * n, n)


def _test():
    s = Solution()
    assert abs(s.getProbability([1, 1]) - 1.0) < 1e-5
    assert abs(s.getProbability([2, 1, 1]) - 0.6666666666666666) < 1e-5
    assert abs(s.getProbability([1, 2, 1, 2]) - 0.60000) < 1e-5
    print("LC 1467 OK")


if __name__ == "__main__":
    _test()
