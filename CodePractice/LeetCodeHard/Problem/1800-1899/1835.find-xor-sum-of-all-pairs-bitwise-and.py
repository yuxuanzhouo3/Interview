"""
LC 1835. Find XOR Sum of All Pairs Bitwise AND — Hard
https://leetcode.com/problems/find-xor-sum-of-all-pairs-bitwise-and/
Best: (xor arr1) & (xor arr2) — O(n+m)
"""

from typing import List
from functools import reduce
from operator import xor


class Solution:
    def getXORSum(self, arr1: List[int], arr2: List[int]) -> int:
        return reduce(xor, arr1) & reduce(xor, arr2)


def _test():
    s = Solution()
    assert s.getXORSum([1, 2, 3], [6, 5]) == 0
    assert s.getXORSum([12], [4]) == 4
    print("LC 1835 OK")


if __name__ == "__main__":
    _test()
