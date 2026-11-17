"""
LC 1728. Cat and Mouse II — Hard
https://leetcode.com/problems/cat-and-mouse-ii/
Best: Game DP / memo on (mouse, cat, turns) — O((mn)^2 · turns)
"""

from functools import lru_cache
from typing import List


class Solution:
    def canMouseWin(self, grid: List[str], catJump: int, mouseJump: int) -> bool:
        m, n = len(grid), len(grid[0])
        walls = set()
        food = mouse = cat = None
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "#":
                    walls.add((i, j))
                elif grid[i][j] == "F":
                    food = (i, j)
                elif grid[i][j] == "M":
                    mouse = (i, j)
                elif grid[i][j] == "C":
                    cat = (i, j)
        dirs = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        # turns limit: 1000 but state space (mn)^2 * 2; mouse wins if food before catch
        # use turn cap = m*n*2 enough by theory often 1000
        MAX_TURN = m * n * 2

        def jumps(pos, jump):
            r, c = pos
            res = [(r, c)]
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                for k in range(1, jump + 1):
                    nr, nc = r + dr * k, c + dc * k
                    if not (0 <= nr < m and 0 <= nc < n) or (nr, nc) in walls:
                        break
                    res.append((nr, nc))
            return res

        @lru_cache(None)
        def win(mr, mc, cr, cc, turn):
            if turn > MAX_TURN:
                return False
            if (mr, mc) == food:
                return True
            if (cr, cc) == food or (mr, mc) == (cr, cc):
                return False
            mouse_turn = turn % 2 == 0
            if mouse_turn:
                for nr, nc in jumps((mr, mc), mouseJump):
                    if win(nr, nc, cr, cc, turn + 1):
                        return True
                return False
            else:
                for nr, nc in jumps((cr, cc), catJump):
                    if not win(mr, mc, nr, nc, turn + 1):
                        return False
                return True

        return win(mouse[0], mouse[1], cat[0], cat[1], 0)


def _test():
    s = Solution()
    assert s.canMouseWin(["####F", "#C...", "M...."], 1, 2) is True
    assert s.canMouseWin(["M.C...F"], 1, 4) is True
    assert s.canMouseWin(["M.C...F"], 1, 3) is False
    print("LC 1728 OK")


if __name__ == "__main__":
    _test()
