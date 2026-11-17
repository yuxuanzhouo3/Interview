"""
LC 2123. Minimum Operations to Remove Adjacent Ones in Matrix — Hard
https://leetcode.com/problems/minimum-operations-to-remove-adjacent-ones-in-matrix/
Best: 二分图最大匹配 = 最小点覆盖 — 匈牙利
"""

from typing import List
from collections import defaultdict


class Solution:
    def minimumOperations(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        g = defaultdict(list)
        for i in range(m):
            for j in range(n):
                if grid[i][j] and (i + j) % 2:
                    x = i * n + j
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < m and 0 <= nj < n and grid[ni][nj]:
                            g[x].append(ni * n + nj)

        match = [-1] * (m * n)

        def find(i, vis):
            for j in g[i]:
                if j in vis:
                    continue
                vis.add(j)
                if match[j] == -1 or find(match[j], vis):
                    match[j] = i
                    return 1
            return 0

        return sum(find(i, set()) for i in g)


def _test():
    s = Solution()
    assert s.minimumOperations([[1, 1, 0], [0, 1, 1], [1, 1, 1]]) == 3
    assert s.minimumOperations([[0, 1], [1, 0]]) == 0
    assert s.minimumOperations([[1, 1], [1, 1]]) == 2
    print("LC 2123 OK")


if __name__ == "__main__":
    _test()
