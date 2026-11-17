"""
LC 2732. Find a Good Subset of the Matrix — Hard
https://leetcode.com/problems/find-a-good-subset-of-the-matrix/
Best: At most 2 rows; bitmask check — O(m·2^n) n≤5
"""

from typing import List


class Solution:
    def goodSubsetofBinaryMatrix(self, grid: List[List[int]]) -> List[int]:
        m, n = len(grid), len(grid[0])
        first = {}
        for i, row in enumerate(grid):
            mask = 0
            for j, v in enumerate(row):
                if v:
                    mask |= 1 << j
            if mask == 0:
                return [i]
            first[mask] = i
        masks = list(first.items())
        for a, i in masks:
            for b, j in masks:
                if a & b == 0:
                    return sorted([i, j])
        return []


def _test():
    s = Solution()
    assert s.goodSubsetofBinaryMatrix([[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]]) == [0, 1]
    assert s.goodSubsetofBinaryMatrix([[0]]) == [0]
    assert s.goodSubsetofBinaryMatrix([[1, 1, 1], [1, 1, 1]]) == []
    print("LC 2732 OK")


if __name__ == "__main__":
    _test()
