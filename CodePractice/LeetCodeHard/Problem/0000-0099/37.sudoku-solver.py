"""
LC 37. Sudoku Solver — Hard
Best: Backtracking + bitmasks / sets — fill empties
"""

from typing import List


class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9
        empties = []

        def bit(d):
            return 1 << (d - 1)

        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    empties.append((i, j))
                    continue
                d = int(board[i][j])
                m = bit(d)
                rows[i] |= m
                cols[j] |= m
                boxes[(i // 3) * 3 + j // 3] |= m

        def dfs(k):
            if k == len(empties):
                return True
            i, j = empties[k]
            b = (i // 3) * 3 + j // 3
            used = rows[i] | cols[j] | boxes[b]
            for d in range(1, 10):
                m = bit(d)
                if used & m:
                    continue
                board[i][j] = str(d)
                rows[i] |= m
                cols[j] |= m
                boxes[b] |= m
                if dfs(k + 1):
                    return True
                rows[i] ^= m
                cols[j] ^= m
                boxes[b] ^= m
                board[i][j] = "."
            return False

        dfs(0)


def _test():
    board = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"],
    ]
    Solution().solveSudoku(board)
    assert board[0][2] == "4" and board[8][8] == "9"
    print("LC 37 OK")


if __name__ == "__main__":
    _test()
