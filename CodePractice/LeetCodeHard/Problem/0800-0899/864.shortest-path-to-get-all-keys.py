"""
LC 864. Shortest Path to Get All Keys — Hard
Best: BFS (i,j,keymask) — O(mn * 2^k)
"""

from typing import List
from collections import deque


class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        m, n = len(grid), len(grid[0])
        keys = 0
        sr = sc = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "@":
                    sr, sc = i, j
                elif "a" <= grid[i][j] <= "f":
                    keys |= 1 << (ord(grid[i][j]) - 97)
        q = deque([(sr, sc, 0, 0)])
        seen = {(sr, sc, 0)}
        while q:
            i, j, mask, d = q.popleft()
            if mask == keys:
                return d
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if not (0 <= ni < m and 0 <= nj < n):
                    continue
                c = grid[ni][nj]
                if c == "#":
                    continue
                nmask = mask
                if "a" <= c <= "f":
                    nmask |= 1 << (ord(c) - 97)
                if "A" <= c <= "F" and not (mask & (1 << (ord(c) - 65))):
                    continue
                state = (ni, nj, nmask)
                if state not in seen:
                    seen.add(state)
                    q.append((ni, nj, nmask, d + 1))
        return -1


def _test():
    s = Solution()
    assert s.shortestPathAllKeys(["@.a.#", "###.#", "b.A.B"]) == 8
    print("LC 864 OK")


if __name__ == "__main__":
    _test()
