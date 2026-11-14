"""
LC 827. Making A Large Island — Hard
Best: Label islands + try flip 0 — O(n^2)
"""

from typing import List


class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        n = len(grid)
        area = {}
        island_id = 2

        def dfs(i, j, iid):
            if i < 0 or i >= n or j < 0 or j >= n or grid[i][j] != 1:
                return 0
            grid[i][j] = iid
            return 1 + dfs(i + 1, j, iid) + dfs(i - 1, j, iid) + dfs(i, j + 1, iid) + dfs(i, j - 1, iid)

        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    area[island_id] = dfs(i, j, island_id)
                    island_id += 1
        ans = max(area.values() or [0])
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    seen = set()
                    cur = 1
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] > 1 and grid[ni][nj] not in seen:
                            seen.add(grid[ni][nj])
                            cur += area[grid[ni][nj]]
                    ans = max(ans, cur)
        return ans


def _test():
    s = Solution()
    assert s.largestIsland([[1, 0], [0, 1]]) == 3
    assert s.largestIsland([[1, 1], [1, 0]]) == 4
    print("LC 827 OK")


if __name__ == "__main__":
    _test()
