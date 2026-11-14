"""
LC 296. Best Meeting Point — Hard
https://leetcode.com/problems/best-meeting-point/
Best: 曼哈顿可分维，中位数最优 — O(mn)
"""

from typing import List


class Solution:
    def minTotalDistance(self, grid: List[List[int]]) -> int:
        rows, cols = [], []
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    rows.append(i)
        for j in range(n):
            for i in range(m):
                if grid[i][j] == 1:
                    cols.append(j)

        def dist_to_median(arr: List[int]) -> int:
            med = arr[len(arr) // 2]
            return sum(abs(x - med) for x in arr)

        return dist_to_median(rows) + dist_to_median(cols)


def _test():
    s = Solution()
    assert s.minTotalDistance([[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]) == 6
    assert s.minTotalDistance([[1, 1]]) == 1
    assert s.minTotalDistance([[1]]) == 0
    print("LC 296 OK")


if __name__ == "__main__":
    _test()
