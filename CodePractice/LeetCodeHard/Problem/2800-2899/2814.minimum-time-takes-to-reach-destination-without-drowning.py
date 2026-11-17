"""
LC 2814. Minimum Time Takes to Reach Destination Without Drowning — Hard
Best: Multi-source water BFS then person BFS — O(nm)
"""

from typing import List
from collections import deque


class Solution:
    def minimumSeconds(self, land: List[List[str]]) -> int:
        m, n = len(land), len(land[0])
        INF = 10**9
        water = [[INF] * n for _ in range(m)]
        q = deque()
        start = dest = None
        for i in range(m):
            for j in range(n):
                if land[i][j] == "*":
                    water[i][j] = 0
                    q.append((i, j))
                elif land[i][j] == "S":
                    start = (i, j)
                elif land[i][j] == "D":
                    dest = (i, j)
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while q:
            x, y = q.popleft()
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and land[nx][ny] not in ("X", "D") and water[nx][ny] > water[x][y] + 1:
                    water[nx][ny] = water[x][y] + 1
                    q.append((nx, ny))
        q = deque([(start[0], start[1], 0)])
        seen = {start}
        while q:
            x, y, t = q.popleft()
            if (x, y) == dest:
                return t
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in seen and land[nx][ny] != "X":
                    nt = t + 1
                    if land[nx][ny] == "D" or nt < water[nx][ny]:
                        seen.add((nx, ny))
                        q.append((nx, ny, nt))
        return -1


def _test():
    s = Solution()
    assert s.minimumSeconds([["D", ".", "*"], [".", ".", "."], [".", "S", "."]]) == 3
    print("LC 2814 OK")


if __name__ == "__main__":
    _test()
