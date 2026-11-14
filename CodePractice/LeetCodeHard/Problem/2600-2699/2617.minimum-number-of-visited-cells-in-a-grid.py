"""
LC 2617. Minimum Number of Visited Cells in a Grid — Hard
https://leetcode.com/problems/minimum-number-of-visited-cells-in-a-grid/
Best: BFS + union-find skip visited on row/col — O(mn α)
"""

from typing import List
from collections import deque


class Solution:
    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        if m == 1 and n == 1:
            return 1
        dist = [[-1] * n for _ in range(m)]
        dist[0][0] = 1
        next_col = [list(range(n + 1)) for _ in range(m)]
        next_row = [list(range(m + 1)) for _ in range(n)]

        def find(fa, x):
            while fa[x] != x:
                fa[x] = fa[fa[x]]
                x = fa[x]
            return x

        q = deque([(0, 0)])
        while q:
            i, j = q.popleft()
            lim = min(n - 1, j + grid[i][j])
            c = find(next_col[i], j + 1)
            while c <= lim:
                if dist[i][c] < 0:
                    dist[i][c] = dist[i][j] + 1
                    if i == m - 1 and c == n - 1:
                        return dist[i][c]
                    q.append((i, c))
                next_col[i][c] = find(next_col[i], c + 1)
                c = find(next_col[i], c)
            limr = min(m - 1, i + grid[i][j])
            r = find(next_row[j], i + 1)
            while r <= limr:
                if dist[r][j] < 0:
                    dist[r][j] = dist[i][j] + 1
                    if r == m - 1 and j == n - 1:
                        return dist[r][j]
                    q.append((r, j))
                next_row[j][r] = find(next_row[j], r + 1)
                r = find(next_row[j], r)
        return -1


def _test():
    s = Solution()
    assert s.minimumVisitedCells([[3, 4, 2, 1], [4, 2, 3, 1], [2, 1, 0, 0], [2, 4, 0, 0]]) == 4
    assert s.minimumVisitedCells([[3, 4, 2, 1], [4, 2, 1, 1], [2, 1, 1, 0], [3, 4, 1, 0]]) == 3
    assert s.minimumVisitedCells([[2, 1, 0], [1, 0, 0]]) == -1
    print("LC 2617 OK")


if __name__ == "__main__":
    _test()
