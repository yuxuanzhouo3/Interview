"""
LC 803. Bricks Falling When Hit — Hard
Best: Reverse hits + Union-Find to roof — O(n*alpha)
"""

from typing import List


class Solution:
    def hitBricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
        m, n = len(grid), len(grid[0])
        parent = list(range(m * n + 1))
        size = [1] * (m * n + 1)
        ROOF = m * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            parent[ra] = rb
            size[rb] += size[ra]

        def idx(i, j):
            return i * n + j

        # copy and erase hits
        g = [row[:] for row in grid]
        for x, y in hits:
            g[x][y] = 0

        for i in range(m):
            for j in range(n):
                if g[i][j] == 0:
                    continue
                if i == 0:
                    union(idx(i, j), ROOF)
                for di, dj in ((1, 0), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n and g[ni][nj]:
                        union(idx(i, j), idx(ni, nj))

        ans = [0] * len(hits)
        for k in range(len(hits) - 1, -1, -1):
            x, y = hits[k]
            if grid[x][y] == 0:
                continue
            pre = size[find(ROOF)]
            g[x][y] = 1
            if x == 0:
                union(idx(x, y), ROOF)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = x + di, y + dj
                if 0 <= ni < m and 0 <= nj < n and g[ni][nj]:
                    union(idx(x, y), idx(ni, nj))
            after = size[find(ROOF)]
            ans[k] = max(0, after - pre - 1)
        return ans


def _test():
    s = Solution()
    assert s.hitBricks([[1, 0, 0, 0], [1, 1, 1, 0]], [[1, 0]]) == [2]
    print("LC 803 OK")


if __name__ == "__main__":
    _test()
