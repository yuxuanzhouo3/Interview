"""
LC 3459. Length of Longest V-Shaped Diagonal Segment — Hard
https://leetcode.com/problems/length-of-longest-v-shaped-diagonal-segment/
Best: DFS/DP memo on direction + turned — O(n*m)
"""

from typing import List
from functools import lru_cache


class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        # diagonals: NE, SE, SW, NW
        dirs = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        # turn clockwise: 0->1->2->3->0
        next_val = {1: 2, 2: 0, 0: 2}

        @lru_cache(None)
        def dfs(i, j, d, turned, expect):
            ni, nj = i + dirs[d][0], j + dirs[d][1]
            if not (0 <= ni < n and 0 <= nj < m) or grid[ni][nj] != expect:
                return 0
            # continue straight
            best = 1 + dfs(ni, nj, d, turned, next_val[expect])
            if not turned:
                nd = (d + 1) % 4  # clockwise turn
                best = max(best, 1 + dfs(ni, nj, nd, True, next_val[expect]))
            return best

        ans = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    for d in range(4):
                        ans = max(ans, 1 + dfs(i, j, d, False, 2))
        return ans


def _test():
    s = Solution()
    assert s.lenOfVDiagonal([[2, 2, 1, 2, 2], [2, 0, 2, 2, 0], [2, 0, 1, 1, 0], [1, 0, 2, 2, 2], [2, 0, 0, 2, 2]]) == 5
    assert s.lenOfVDiagonal([[2, 2, 2, 2, 2], [2, 0, 2, 2, 0], [2, 0, 1, 1, 0], [1, 0, 2, 2, 2], [2, 0, 0, 2, 2]]) == 4
    assert s.lenOfVDiagonal([[1]]) == 1
    print("LC 3459 OK")


if __name__ == "__main__":
    _test()
