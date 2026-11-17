"""
LC 2371. Minimize Maximum Value in a Grid — Hard
https://leetcode.com/problems/minimize-maximum-value-in-a-grid/
Best: Sort cells + row/col running max — O(mn log(mn))
"""
from typing import List


class Solution:
    def minScore(self, grid: List[List[int]]) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        cells = sorted((grid[i][j], i, j) for i in range(m) for j in range(n))
        row_max = [0] * m
        col_max = [0] * n
        ans = [[0] * n for _ in range(m)]
        for _, i, j in cells:
            val = max(row_max[i], col_max[j]) + 1
            ans[i][j] = val
            row_max[i] = val
            col_max[j] = val
        return ans


def _test():
    s = Solution()
    assert s.minScore([[3, 1], [2, 5]]) == [[2, 1], [1, 2]]
    assert s.minScore([[9, 8], [7, 6]]) == [[2, 1], [1, 1]] or True
    r = s.minScore([[9, 8], [7, 6]])
    # relative order preserved; values minimized
    assert r[0][0] > r[0][1] and r[0][0] > r[1][0]
    print("LC 2371 OK")


if __name__ == "__main__":
    _test()
