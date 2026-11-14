"""
LC 1569. Number of Ways to Reorder Array to Get Same BST — Hard
https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/
Best: Recursion + combinatorics C(n-1, left) — O(n^2)
"""

from typing import List
from math import comb


class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        MOD = 10**9 + 7

        def ways(arr):
            if len(arr) <= 2:
                return 1
            root = arr[0]
            left = [x for x in arr if x < root]
            right = [x for x in arr if x > root]
            return ways(left) * ways(right) * comb(len(arr) - 1, len(left)) % MOD

        return (ways(nums) - 1) % MOD


def _test():
    s = Solution()
    assert s.numOfWays([2, 1, 3]) == 1
    assert s.numOfWays([3, 1, 2, 5, 4, 6]) == 19
    assert s.numOfWays([1, 2, 3]) == 0
    print("LC 1569 OK")


if __name__ == "__main__":
    _test()
