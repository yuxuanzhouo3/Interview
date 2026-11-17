"""
LC 1250. Check If It Is a Good Array — Hard
https://leetcode.com/problems/check-if-it-is-a-good-array/
Best: Bezout — gcd of all == 1 — O(n log A)
"""

from math import gcd
from typing import List


class Solution:
    def isGoodArray(self, nums: List[int]) -> bool:
        g = 0
        for x in nums:
            g = gcd(g, x)
            if g == 1:
                return True
        return g == 1


def _test():
    s = Solution()
    assert s.isGoodArray([12, 5, 7, 23]) is True
    assert s.isGoodArray([29, 6, 10]) is True
    assert s.isGoodArray([3, 6]) is False
    print("LC 1250 OK")


if __name__ == "__main__":
    _test()
