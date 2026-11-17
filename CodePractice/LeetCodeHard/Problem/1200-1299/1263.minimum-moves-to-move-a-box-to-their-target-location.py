"""
LC 1263. Minimum Moves to Move a Box to Their Target Location — Hard
https://leetcode.com/problems/minimum-moves-to-move-a-box-to-their-target-location/
Best: BFS on box pushes; inner BFS for player reachability — O((mn)^2)
"""

from collections import deque
from typing import List


class Solution:
    def minPushBox(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        br = bc = pr = pc = tr = tc = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "B":
                    br, bc = i, j
                elif grid[i][j] == "S":
                    pr, pc = i, j
                elif grid[i][j] == "T":
                    tr, tc = i, j

        def valid(r, c):
            return 0 <= r < m and 0 <= c < n and grid[r][c] != "#"

        def can_reach(sr, sc, er, ec, box_r, box_c):
            if (sr, sc) == (er, ec):
                return True
            q = deque([(sr, sc)])
            seen = {(sr, sc)}
            while q:
                r, c = q.popleft()
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = r + dr, c + dc
                    if not valid(nr, nc) or (nr, nc) in seen or (nr, nc) == (box_r, box_c):
                        continue
                    if (nr, nc) == (er, ec):
                        return True
                    seen.add((nr, nc))
                    q.append((nr, nc))
            return False

        q = deque([(br, bc, pr, pc, 0)])
        seen = {(br, bc, pr, pc)}
        while q:
            b_r, b_c, p_r, p_c, pushes = q.popleft()
            if (b_r, b_c) == (tr, tc):
                return pushes
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb_r, nb_c = b_r + dr, b_c + dc
                stand_r, stand_c = b_r - dr, b_c - dc
                if not valid(nb_r, nb_c) or not valid(stand_r, stand_c):
                    continue
                if not can_reach(p_r, p_c, stand_r, stand_c, b_r, b_c):
                    continue
                state = (nb_r, nb_c, b_r, b_c)  # player ends at old box
                if state in seen:
                    continue
                seen.add(state)
                q.append((nb_r, nb_c, b_r, b_c, pushes + 1))
        return -1


def _test():
    s = Solution()
    grid = [
        ["#", "#", "#", "#", "#", "#"],
        ["#", "T", "#", "#", "#", "#"],
        ["#", ".", ".", "B", ".", "#"],
        ["#", ".", "#", "#", ".", "#"],
        ["#", ".", ".", ".", "S", "#"],
        ["#", "#", "#", "#", "#", "#"],
    ]
    assert s.minPushBox(grid) == 3
    grid2 = [
        ["#", "#", "#", "#", "#", "#"],
        ["#", "T", "#", "#", "#", "#"],
        ["#", ".", ".", "B", ".", "#"],
        ["#", "#", "#", "#", ".", "#"],
        ["#", ".", ".", ".", "S", "#"],
        ["#", "#", "#", "#", "#", "#"],
    ]
    assert s.minPushBox(grid2) == -1
    print("LC 1263 OK")


if __name__ == "__main__":
    _test()
