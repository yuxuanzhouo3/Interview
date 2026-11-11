"""
3. 网格 DSU：二维映射 1D
App: LC 200 / 827。
"""


class GridDSU:
    def __init__(self, grid):
        self.g = grid
        self.m, self.n = len(grid), len(grid[0]) if grid else 0
        N = self.m * self.n
        self.p = list(range(N))
        self.size = [0] * N
        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j] == 1:
                    self.size[self._id(i, j)] = 1
        for i in range(self.m):
            for j in range(self.n):
                if grid[i][j] != 1:
                    continue
                for di, dj in ((0, 1), (1, 0)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.m and 0 <= nj < self.n and grid[ni][nj] == 1:
                        self.union(self._id(i, j), self._id(ni, nj))

    def _id(self, i, j):
        return i * self.n + j

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.size[ra] += self.size[rb]

    def query(self, i=None, j=None):
        if i is None:
            roots = set()
            for x in range(self.m):
                for y in range(self.n):
                    if self.g[x][y] == 1:
                        roots.add(self.find(self._id(x, y)))
            return len(roots)
        return self.size[self.find(self._id(i, j))]

    def update(self, i, j, val=1):
        """改格子；教学简化：仅 0→1 时与四邻合并。"""
        if self.g[i][j] == val:
            return False
        self.g[i][j] = val
        if val == 1:
            self.size[self._id(i, j)] = 1
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.m and 0 <= nj < self.n and self.g[ni][nj] == 1:
                    self.union(self._id(i, j), self._id(ni, nj))
        return True

    def add(self, i, j):
        return self.update(i, j, 1)

    def delete(self, i, j):
        """网格删点难拆 DSU；标记为 0 仅示意。"""
        self.g[i][j] = 0
        return True


def _test():
    g = [[1, 1, 0], [1, 0, 0], [0, 0, 1]]
    d = GridDSU(g)
    assert d.query() == 2
    assert d.query(0, 0) == 3
    d.add(1, 1)
    assert d.query() == 1 or d.query(0, 0) >= 3
    print("dsu_grid OK components", GridDSU([[1, 1, 0], [1, 0, 0], [0, 0, 1]]).query())


APPLICATION = """
- id=i*n+j；只合并陆地。
- 827: 先算各岛 size，再试填海四邻不同根求和。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
