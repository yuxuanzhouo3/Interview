"""
LC 51. N-Queens — Hard
Best: Backtracking with col/diag sets — O(n!)
"""

from typing import List


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        cols, d1, d2 = set(), set(), set()
        board = [["."] * n for _ in range(n)]
        ans = []

        def dfs(r):
            if r == n:
                ans.append(["".join(row) for row in board])
                return
            for c in range(n):
                if c in cols or r - c in d1 or r + c in d2:
                    continue
                cols.add(c); d1.add(r - c); d2.add(r + c)
                board[r][c] = "Q"
                dfs(r + 1)
                board[r][c] = "."
                cols.remove(c); d1.remove(r - c); d2.remove(r + c)

        dfs(0)
        return ans


def _test():
    s = Solution()
    assert len(s.solveNQueens(4)) == 2
    assert len(s.solveNQueens(1)) == 1
    print("LC 51 OK")


if __name__ == "__main__":
    _test()
