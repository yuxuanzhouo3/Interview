"""
LC 1368. Minimum Cost to Make at Least One Valid Path in a Grid — Hard
https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/
Best: 0-1 BFS — O(mn)
"""

from typing import List
from collections import deque


class Solution:
    def minCost(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 1,2,3,4
        dist = [[10**9] * n for _ in range(m)]
        dist[0][0] = 0
        dq = deque([(0, 0)])
        while dq:
            r, c = dq.popleft()
            if (r, c) == (m - 1, n - 1):
                return dist[r][c]
            for i, (dr, dc) in enumerate(dirs):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    w = 0 if grid[r][c] == i + 1 else 1
                    nd = dist[r][c] + w
                    if nd < dist[nr][nc]:
                        dist[nr][nc] = nd
                        if w == 0:
                            dq.appendleft((nr, nc))
                        else:
                            dq.append((nr, nc))
        return dist[m - 1][n - 1]


def _test():
    s = Solution()
    assert s.minCost([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1], [2, 2, 2, 2]]) == 3
    assert s.minCost([[1, 1, 3], [3, 2, 2], [1, 1, 4]]) == 0
    assert s.minCost([[1, 2], [4, 3]]) == 1
    print("LC 1368 OK")


if __name__ == "__main__":
    _test()
