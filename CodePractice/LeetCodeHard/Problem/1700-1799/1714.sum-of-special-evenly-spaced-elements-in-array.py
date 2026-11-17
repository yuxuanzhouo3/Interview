"""
LC 1714. Sum Of Special Evenly-Spaced Elements In Array — Hard
https://leetcode.com/problems/sum-of-special-evenly-spaced-elements-in-array/
Best: 分块：小步长预处理后缀，大步长暴力 — O((n+q)√n)
"""

from typing import List
from math import isqrt


class Solution:
    def solve(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        MOD = 10**9 + 7
        n = len(nums)
        m = isqrt(n)
        suf = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(n - 1, -1, -1):
                suf[i][j] = suf[i][min(n, j + i)] + nums[j]
        ans = []
        for x, y in queries:
            if y <= m:
                ans.append(suf[y][x] % MOD)
            else:
                ans.append(sum(nums[x::y]) % MOD)
        return ans


def _test():
    s = Solution()
    assert s.solve([0, 1, 2, 3, 4, 5, 6, 7], [[0, 3], [5, 1], [4, 2]]) == [9, 18, 10]
    assert s.solve([100, 200, 101, 201, 102, 202, 103, 203], [[0, 7]]) == [303]
    print("LC 1714 OK")


if __name__ == "__main__":
    _test()
