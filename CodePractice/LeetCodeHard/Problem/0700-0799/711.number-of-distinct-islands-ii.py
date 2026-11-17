"""
LC 711. Number of Distinct Islands II — Hard
https://leetcode.com/problems/number-of-distinct-islands-ii/
Tags: DFS, Hash, Geometry
Best: DFS 取点集 + 8 种变换规范化
"""

from typing import List


class Solution:
    def numDistinctIslands2(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        m, n = len(grid), len(grid[0])
        seen = set()

        def dfs(i, j, cells):
            if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] != 1:
                return
            grid[i][j] = 0
            cells.append((i, j))
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                dfs(i + di, j + dj, cells)

        def normalize(cells):
            shapes = []
            for s, t in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                for swap in (False, True):
                    pts = []
                    for x, y in cells:
                        a, b = (y, x) if swap else (x, y)
                        pts.append((a * s, b * t))
                    pts.sort()
                    ox, oy = pts[0]
                    shapes.append(tuple((x - ox, y - oy) for x, y in pts))
            return min(shapes)

        shapes = set()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    cells = []
                    dfs(i, j, cells)
                    shapes.add(normalize(cells))
        return len(shapes)


def _test():
    sol = Solution()
    g1 = [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]
    assert sol.numDistinctIslands2([row[:] for row in g1]) == 1
    g2 = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]
    assert sol.numDistinctIslands2([row[:] for row in g2]) == 1
    print("LC 711 OK")


if __name__ == "__main__":
    _test()

