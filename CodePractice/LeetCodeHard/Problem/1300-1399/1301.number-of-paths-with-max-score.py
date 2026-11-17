"""
LC 1301. Number of Paths with Max Score — Hard
https://leetcode.com/problems/number-of-paths-with-max-score/
Best: DP from bottom-right — O(n^2)
"""

from typing import List


class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        MOD = 10**9 + 7
        n = len(board)
        b = [list(row) for row in board]
        INF = 10**18
        score = [[-INF] * n for _ in range(n)]
        ways = [[0] * n for _ in range(n)]
        score[n - 1][n - 1] = 0
        ways[n - 1][n - 1] = 1
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if b[i][j] == "X" or (i == n - 1 and j == n - 1):
                    continue
                if ways[i][j] == 0 and score[i][j] == -INF:
                    # will be set from predecessors — init via neighbors below/right
                    pass
                best = -INF
                w = 0
                for di, dj in ((1, 0), (0, 1), (1, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and ways[ni][nj]:
                        if score[ni][nj] > best:
                            best = score[ni][nj]
                            w = ways[ni][nj]
                        elif score[ni][nj] == best:
                            w = (w + ways[ni][nj]) % MOD
                if w == 0:
                    continue
                add = 0 if b[i][j] in "SE" else int(b[i][j])
                score[i][j] = best + add
                ways[i][j] = w
        if ways[0][0] == 0:
            return [0, 0]
        return [score[0][0], ways[0][0]]


def _test():
    s = Solution()
    assert s.pathsWithMaxScore(["E23", "2X2", "12S"]) == [7, 1]
    assert s.pathsWithMaxScore(["E12", "1X1", "21S"]) == [4, 2]
    assert s.pathsWithMaxScore(["E11", "XXX", "11S"]) == [0, 0]
    print("LC 1301 OK")


if __name__ == "__main__":
    _test()
