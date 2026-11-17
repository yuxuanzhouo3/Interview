"""
LC 2056. Number of Valid Move Combinations On Chessboard — Hard
https://leetcode.com/problems/number-of-valid-move-combinations-on-chessboard/
Best: 枚举每子目标/路径 + 冲突检测 — 小状态
"""

from typing import List
from itertools import product


class Solution:
    def countCombinations(self, pieces: List[str], positions: List[List[int]]) -> int:
        dirs = {
            "rook": [(1, 0), (-1, 0), (0, 1), (0, -1)],
            "bishop": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            "queen": [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
        }
        n = len(pieces)
        all_moves = []  # list of list of (seq of positions including start)

        def gen(piece, r, c):
            moves = [[(r, c)]]  # stay
            for dr, dc in dirs[piece]:
                path = [(r, c)]
                nr, nc = r + dr, c + dc
                while 1 <= nr <= 8 and 1 <= nc <= 8:
                    path.append((nr, nc))
                    moves.append(path[:])
                    nr += dr
                    nc += dc
            return moves

        for p, (r, c) in zip(pieces, positions):
            all_moves.append(gen(p, r, c))

        ans = 0
        for combo in product(*all_moves):
            # simulate until all stopped
            max_t = max(len(m) for m in combo)
            ok = True
            for t in range(max_t):
                cells = []
                for m in combo:
                    cells.append(m[t] if t < len(m) else m[-1])
                if len(set(cells)) < n:
                    ok = False
                    break
            if ok:
                ans += 1
        return ans


def _test():
    s = Solution()
    assert s.countCombinations(["rook"], [[1, 1]]) == 15
    assert s.countCombinations(["queen"], [[1, 1]]) == 22
    assert s.countCombinations(["bishop"], [[4, 3]]) == 12
    assert s.countCombinations(["rook", "rook"], [[1, 1], [8, 8]]) == 223
    print("LC 2056 OK")


if __name__ == "__main__":
    _test()
