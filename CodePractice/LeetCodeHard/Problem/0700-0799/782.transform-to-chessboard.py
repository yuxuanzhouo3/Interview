"""
LC 782. Transform to Chessboard — Hard
Best: Row/col pattern count + moves — O(n^2)
"""

from typing import List
from collections import Counter


class Solution:
    def movesToChessboard(self, board: List[List[int]]) -> int:
        n = len(board)
        rows = [tuple(r) for r in board]
        cnt = Counter(rows)
        if len(cnt) != 2:
            return -1
        (r1, c1), (r2, c2) = cnt.most_common()
        if abs(c1 - c2) > 1:
            return -1
        if any(a == b for a, b in zip(r1, r2)):
            return -1
        # columns similarly implied by rows
        def moves(line0):
            # count mismatches to 0101... and 1010...
            n = len(line0)
            ones = sum(line0)
            if abs(ones * 2 - n) > 1:
                return -1
            diff0 = sum(line0[i] != i % 2 for i in range(n))
            diff1 = sum(line0[i] != (i + 1) % 2 for i in range(n))
            if n % 2:
                return diff0 // 2 if ones * 2 > n else diff1 // 2
            return min(diff0, diff1) // 2

        row_moves = moves([board[i][0] for i in range(n)])
        col_moves = moves(board[0][:])
        if row_moves < 0 or col_moves < 0:
            return -1
        return row_moves + col_moves


def _test():
    s = Solution()
    assert s.movesToChessboard([[0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]]) == 2
    assert s.movesToChessboard([[0, 1], [1, 0]]) == 0
    print("LC 782 OK")


if __name__ == "__main__":
    _test()
