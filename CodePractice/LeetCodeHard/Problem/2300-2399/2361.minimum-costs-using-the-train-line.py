"""
LC 2361. Minimum Costs Using the Train Line — Hard
https://leetcode.com/problems/minimum-costs-using-the-train-line/
Best: DP regular/express — O(n)
"""
from typing import List
from math import inf


class Solution:
    def minimumCosts(self, regular: List[int], express: List[int], expressCost: int) -> List[int]:
        n = len(regular)
        f, g = 0, inf
        cost = [0] * n
        for i, (a, b) in enumerate(zip(regular, express)):
            ff = min(f + a, g + a)
            gg = min(f + expressCost + b, g + b)
            f, g = ff, gg
            cost[i] = min(f, g)
        return cost


def _test():
    s = Solution()
    assert s.minimumCosts([1, 6, 9, 5], [5, 2, 3, 10], 8) == [1, 7, 14, 19]
    assert s.minimumCosts([11, 5, 13], [7, 10, 6], 3) == [10, 15, 24]
    print("LC 2361 OK")


if __name__ == "__main__":
    _test()
