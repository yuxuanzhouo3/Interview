"""
LC 1293. Shortest Path in a Grid with Obstacles Elimination — Hard
https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/
Best: BFS state (r,c,remain_k) — O(mnk)
"""

from typing import List
from collections import deque


class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        if k >= m + n - 2:
            return m + n - 2
        seen = [[-1] * n for _ in range(m)]
        seen[0][0] = k
        q = deque([(0, 0, k, 0)])  # r,c,remain,steps
        while q:
            r, c, rem, steps = q.popleft()
            if (r, c) == (m - 1, n - 1):
                return steps
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < m and 0 <= nc < n:
                    nrem = rem - grid[nr][nc]
                    if nrem >= 0 and nrem > seen[nr][nc]:
                        seen[nr][nc] = nrem
                        q.append((nr, nc, nrem, steps + 1))
        return -1


def _test():
    s = Solution()
    assert s.shortestPath([[0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]], 1) == 6
    assert s.shortestPath([[0, 1, 1], [1, 1, 1], [1, 0, 0]], 1) == -1
    print("LC 1293 OK")


if __name__ == "__main__":
    _test()
