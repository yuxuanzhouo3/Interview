"""
LC 1001. Grid Illumination — Hard
https://leetcode.com/problems/grid-illumination/
Best: Hash counts for rows/cols/diags — O(L + Q)
"""

from typing import List
from collections import defaultdict


class Solution:
    def gridIllumination(self, n: int, lamps: List[List[int]], queries: List[List[int]]) -> List[int]:
        rows = defaultdict(int)
        cols = defaultdict(int)
        diag = defaultdict(int)   # r-c
        antidiag = defaultdict(int)  # r+c
        on = set()
        for r, c in lamps:
            if (r, c) in on:
                continue
            on.add((r, c))
            rows[r] += 1
            cols[c] += 1
            diag[r - c] += 1
            antidiag[r + c] += 1

        ans = []
        dirs = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for qr, qc in queries:
            lit = int(rows[qr] > 0 or cols[qc] > 0 or diag[qr - qc] > 0 or antidiag[qr + qc] > 0)
            ans.append(lit)
            for dr, dc in dirs:
                r, c = qr + dr, qc + dc
                if (r, c) in on:
                    on.remove((r, c))
                    rows[r] -= 1
                    cols[c] -= 1
                    diag[r - c] -= 1
                    antidiag[r + c] -= 1
        return ans


def _test():
    s = Solution()
    assert s.gridIllumination(5, [[0, 0], [4, 4]], [[1, 1], [1, 0]]) == [1, 0]
    assert s.gridIllumination(5, [[0, 0], [4, 4]], [[1, 1], [1, 1]]) == [1, 1]
    assert s.gridIllumination(1, [[0, 0]], [[0, 0]]) == [1]
    print("LC 1001 OK")


if __name__ == "__main__":
    _test()
