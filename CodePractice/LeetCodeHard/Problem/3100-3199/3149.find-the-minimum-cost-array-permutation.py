"""
LC 3149. Find the Minimum Cost Array Permutation — Hard
https://leetcode.com/problems/find-the-minimum-cost-array-permutation/
Best: Bitmask DP + reconstruct — O(n^2 2^n)
"""

from functools import cache
from math import inf
from typing import List


class Solution:
    def findPermutation(self, nums: List[int]) -> List[int]:
        n = len(nums)

        @cache
        def dfs(mask: int, pre: int) -> int:
            if mask == (1 << n) - 1:
                return abs(pre - nums[0])
            res = inf
            for cur in range(1, n):
                if (mask >> cur) & 1 == 0:
                    res = min(res, abs(pre - nums[cur]) + dfs(mask | (1 << cur), cur))
            return res

        ans = []

        def g(mask: int, pre: int) -> None:
            ans.append(pre)
            if mask == (1 << n) - 1:
                return
            res = dfs(mask, pre)
            for cur in range(1, n):
                if (mask >> cur) & 1 == 0:
                    if abs(pre - nums[cur]) + dfs(mask | (1 << cur), cur) == res:
                        g(mask | (1 << cur), cur)
                        break

        g(1, 0)
        return ans


def _test():
    s = Solution()
    assert s.findPermutation([1, 0, 2]) == [0, 1, 2]
    assert s.findPermutation([0, 2, 1]) == [0, 2, 1]
    print("LC 3149 OK")


if __name__ == "__main__":
    _test()
