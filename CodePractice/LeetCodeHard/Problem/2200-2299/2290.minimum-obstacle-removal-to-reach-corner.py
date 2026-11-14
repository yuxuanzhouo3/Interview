"""
LC 2290. Minimum Obstacle Removal to Reach Corner — Hard
https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/
Best: 0-1 BFS (deque) — O(mn)
"""

from typing import List
from collections import deque


class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dist = [[10**9] * n for _ in range(m)]
        dist[0][0] = 0
        dq = deque([(0, 0)])
        while dq:
            i, j = dq.popleft()
            if i == m - 1 and j == n - 1:
                return dist[i][j]
            for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= ni < m and 0 <= nj < n:
                    nd = dist[i][j] + grid[ni][nj]
                    if nd < dist[ni][nj]:
                        dist[ni][nj] = nd
                        if grid[ni][nj] == 0:
                            dq.appendleft((ni, nj))
                        else:
                            dq.append((ni, nj))
        return dist[m - 1][n - 1]


def _test():
    s = Solution()
    assert s.minimumObstacles([[0, 1, 1], [1, 1, 0], [1, 1, 0]]) == 2
    assert s.minimumObstacles([[0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]]) == 0
    print("LC 2290 OK")


if __name__ == "__main__":
    _test()
