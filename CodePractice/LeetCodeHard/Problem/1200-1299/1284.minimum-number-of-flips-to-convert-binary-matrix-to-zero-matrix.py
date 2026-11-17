"""
LC 1284. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix — Hard
https://leetcode.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix/
Best: BFS on bitmask (≤ 2^9) — O(2^{mn}·mn)
"""

from collections import deque
from typing import List


class Solution:
    def minFlips(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        start = 0
        for i in range(m):
            for j in range(n):
                if mat[i][j]:
                    start |= 1 << (i * n + j)

        def flip(state: int, i: int, j: int) -> int:
            for x, y in ((i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= x < m and 0 <= y < n:
                    state ^= 1 << (x * n + y)
            return state

        q = deque([(start, 0)])
        seen = {start}
        while q:
            cur, d = q.popleft()
            if cur == 0:
                return d
            for i in range(m):
                for j in range(n):
                    nxt = flip(cur, i, j)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, d + 1))
        return -1


def _test():
    s = Solution()
    assert s.minFlips([[0, 0], [0, 1]]) == 3
    assert s.minFlips([[0]]) == 0
    assert s.minFlips([[1, 0, 0], [1, 0, 0]]) == -1
    print("LC 1284 OK")


if __name__ == "__main__":
    _test()
