"""
LC 2862. Maximum Element-Sum of a Complete Subset of Indices — Hard
https://leetcode.com/problems/maximum-element-sum-of-a-complete-subset-of-indices/
Best: Square-free kernel grouping — O(n√A)
"""

from typing import List
from math import isqrt


class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # indices 1-based; complete subset <=> all pairwise products are perfect squares
        # <=> all nums[i-1] have same square-free kernel after removing square factors from index
        # Actually: indices i,j complete if i*j is square. Group by square-free part of index.
        best = 0
        n = len(nums)
        from collections import defaultdict
        group = defaultdict(int)
        for i in range(1, n + 1):
            x = i
            sf = 1
            d = 2
            while d * d <= x:
                c = 0
                while x % d == 0:
                    x //= d
                    c += 1
                if c % 2:
                    sf *= d
                d += 1
            if x > 1:
                sf *= x
            group[sf] += nums[i - 1]
        return max(group.values())


def _test():
    s = Solution()
    assert s.maximumSum([8, 7, 3, 5, 7, 2, 4, 9]) == 16
    assert s.maximumSum([5, 10, 3, 10, 1, 13, 7, 9, 4]) == 19
    print("LC 2862 OK")


if __name__ == "__main__":
    _test()

