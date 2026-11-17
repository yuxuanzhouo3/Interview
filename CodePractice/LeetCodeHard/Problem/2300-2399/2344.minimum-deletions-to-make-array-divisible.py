"""
LC 2344. Minimum Deletions to Make Array Divisible — Hard
https://leetcode.com/problems/minimum-deletions-to-make-array-divisible/
Best: GCD of numsDivide + scan min divisor — O(n log A + m log A)
"""

from typing import List
import math


class Solution:
    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
        g = numsDivide[0]
        for x in numsDivide[1:]:
            g = math.gcd(g, x)
        nums.sort()
        for i, x in enumerate(nums):
            if g % x == 0:
                return i
        return -1


def _test():
    s = Solution()
    assert s.minOperations([2, 3, 2, 4, 3], [9, 6, 9, 3, 15]) == 2
    assert s.minOperations([4, 3, 6], [8, 2, 6, 10]) == -1
    print("LC 2344 OK")


if __name__ == "__main__":
    _test()
