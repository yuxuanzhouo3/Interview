"""
LC 1074. Number of Submatrices That Sum to Target — Hard
https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/
Best: Fix left-right cols + 1D prefix hash — O(m^2 n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        m, n = len(matrix), len(matrix[0])
        ans = 0
        for left in range(n):
            row_sum = [0] * m
            for right in range(left, n):
                for i in range(m):
                    row_sum[i] += matrix[i][right]
                pref = 0
                cnt = defaultdict(int)
                cnt[0] = 1
                for x in row_sum:
                    pref += x
                    ans += cnt[pref - target]
                    cnt[pref] += 1
        return ans


def _test():
    s = Solution()
    assert s.numSubmatrixSumTarget([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 0) == 4
    assert s.numSubmatrixSumTarget([[1, -1], [-1, 1]], 0) == 5
    assert s.numSubmatrixSumTarget([[904]], 0) == 0
    print("LC 1074 OK")


if __name__ == "__main__":
    _test()
