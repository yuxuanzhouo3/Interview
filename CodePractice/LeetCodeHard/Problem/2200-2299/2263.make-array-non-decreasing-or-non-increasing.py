"""
LC 2263. Make Array Non-decreasing or Non-increasing — Hard
https://leetcode.com/problems/make-array-non-decreasing-or-non-increasing/
Best: DP on final value (or heap greedy) — O(n * V), V=1000
"""
from typing import List
from math import inf


class Solution:
    def convertArray(self, nums: List[int]) -> int:
        def solve(arr: List[int]) -> int:
            n = len(arr)
            f = [[0] * 1001 for _ in range(n + 1)]
            for i, x in enumerate(arr, 1):
                mi = inf
                for j in range(1001):
                    mi = min(mi, f[i - 1][j])
                    f[i][j] = mi + abs(x - j)
            return min(f[n])

        return min(solve(nums), solve(nums[::-1]))


def _test():
    s = Solution()
    assert s.convertArray([3, 2, 4, 5, 0]) == 4
    assert s.convertArray([2, 2, 3, 4]) == 0
    assert s.convertArray([0]) == 0
    print("LC 2263 OK")


if __name__ == "__main__":
    _test()
