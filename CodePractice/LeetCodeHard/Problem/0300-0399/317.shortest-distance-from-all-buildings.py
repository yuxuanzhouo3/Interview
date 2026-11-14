"""
LC 317. Shortest Distance from All Buildings — Hard
https://leetcode.com/problems/shortest-distance-from-all-buildings/
Best: 从每栋楼 BFS 累加距离 — O(B·mn)
"""

from typing import List
from collections import deque


class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dist = [[0] * n for _ in range(m)]
        reach = [[0] * n for _ in range(m)]
        buildings = sum(cell == 1 for row in grid for cell in row)
        dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))

        for i in range(m):
            for j in range(n):
                if grid[i][j] != 1:
                    continue
                q = deque([(i, j, 0)])
                seen = [[False] * n for _ in range(m)]
                seen[i][j] = True
                while q:
                    r, c, d = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < m
                            and 0 <= nc < n
                            and not seen[nr][nc]
                            and grid[nr][nc] == 0
                        ):
                            seen[nr][nc] = True
                            dist[nr][nc] += d + 1
                            reach[nr][nc] += 1
                            q.append((nr, nc, d + 1))

        best = float("inf")
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and reach[i][j] == buildings:
                    best = min(best, dist[i][j])
        return -1 if best == float("inf") else best


def _test():
    s = Solution()
    grid = [[1, 0, 2, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]
    # 官方题面写 Output:6 为笔误；到 (0,0)/(0,4)/(2,2) 实为 3+3+1=7
    assert s.shortestDistance(grid) == 7
    assert s.shortestDistance([[1, 0], [0, 1]]) == 2
    assert s.shortestDistance([[1]]) == -1
    print("LC 317 OK")


if __name__ == "__main__":
    _test()
