"""
LC 2258. Escape the Spreading Fire — Hard
https://leetcode.com/problems/escape-the-spreading-fire/
Best: binary search wait + dual BFS — O(mn log(mn))
"""

from typing import List
from collections import deque


class Solution:
    def maximumMinutes(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dirs = (-1, 0, 1, 0, -1)
        fire = [[False] * n for _ in range(m)]

        def spread(q: deque) -> deque:
            nq = deque()
            while q:
                i, j = q.popleft()
                for k in range(4):
                    x, y = i + dirs[k], j + dirs[k + 1]
                    if 0 <= x < m and 0 <= y < n and not fire[x][y] and grid[x][y] == 0:
                        fire[x][y] = True
                        nq.append((x, y))
            return nq

        def check(t: int) -> bool:
            for i in range(m):
                for j in range(n):
                    fire[i][j] = False
            q1 = deque()
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1:
                        fire[i][j] = True
                        q1.append((i, j))
            while t and q1:
                q1 = spread(q1)
                t -= 1
            if fire[0][0]:
                return False
            q2 = deque([(0, 0)])
            vis = [[False] * n for _ in range(m)]
            vis[0][0] = True
            while q2:
                for _ in range(len(q2)):
                    i, j = q2.popleft()
                    if fire[i][j]:
                        continue
                    for k in range(4):
                        x, y = i + dirs[k], j + dirs[k + 1]
                        if (
                            0 <= x < m
                            and 0 <= y < n
                            and not vis[x][y]
                            and not fire[x][y]
                            and grid[x][y] == 0
                        ):
                            if x == m - 1 and y == n - 1:
                                return True
                            vis[x][y] = True
                            q2.append((x, y))
                q1 = spread(q1)
            return False

        lo, hi = -1, m * n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if check(mid):
                lo = mid
            else:
                hi = mid - 1
        return 10**9 if lo == m * n else lo


def _test():
    s = Solution()
    assert s.maximumMinutes(
        [
            [0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 1, 0],
            [0, 2, 0, 0, 1, 2, 0],
            [0, 0, 2, 2, 2, 0, 2],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ) == 3
    assert s.maximumMinutes([[0, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 0]]) == -1
    assert s.maximumMinutes([[0, 0, 0], [2, 2, 0], [1, 2, 0]]) == 10**9
    print("LC 2258 OK")


if __name__ == "__main__":
    _test()
