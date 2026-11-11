"""
AdvancedGraph / Grid Graph（网格 BFS + 连通块）
Template: query(dist|components) / update(cell) / add / del
App: 岛屿 LC 200；网格最短路；超市/仓库网格建模。
来源: CodePractice/graph.py §10
"""

from collections import deque


class GridGraph:
    D = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, grid=None):
        self.grid = [row[:] for row in (grid or [["."]])]
        self.n = len(self.grid)
        self.m = len(self.grid[0])

    def add(self, r, c=None, cell="."):
        if c is None:
            while self.n <= r:
                self.grid.append(["."] * self.m)
                self.n += 1
            return
        self._ensure(r, c)
        self.grid[r][c] = cell

    def _ensure(self, r, c):
        while self.n <= r:
            self.grid.append(["."] * self.m)
            self.n += 1
        if c >= self.m:
            for row in self.grid:
                row.extend(["."] * (c + 1 - self.m))
            self.m = c + 1

    def update(self, r, c, cell):
        if not (0 <= r < self.n and 0 <= c < self.m):
            return False
        self.grid[r][c] = cell
        return True

    def delete(self, r, c):
        """设为障碍 *。"""
        return self.update(r, c, "*")

    def query(self, mode="components", start=None):
        if mode == "components":
            return self.components()
        if mode == "bfs":
            return self.bfs(start)
        raise ValueError(mode)

    def bfs(self, start):
        sx, sy = start
        if self.grid[sx][sy] == "*":
            return None
        dist = [[-1] * self.m for _ in range(self.n)]
        dist[sx][sy] = 0
        q = deque([(sx, sy)])
        while q:
            x, y = q.popleft()
            for dx, dy in self.D:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and self.grid[nx][ny] != "*"
                    and dist[nx][ny] == -1
                ):
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))
        return dist

    def components(self):
        vis = [[False] * self.m for _ in range(self.n)]
        comps = []
        for i in range(self.n):
            for j in range(self.m):
                if self.grid[i][j] == "*" or vis[i][j]:
                    continue
                q = deque([(i, j)])
                vis[i][j] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in self.D:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < self.n
                            and 0 <= ny < self.m
                            and self.grid[nx][ny] != "*"
                            and not vis[nx][ny]
                        ):
                            vis[nx][ny] = True
                            q.append((nx, ny))
                comps.append(comp)
        return comps


def _test():
    grid = [
        [".", ".", "*"],
        ["*", "*", "*"],
        [".", ".", "."],
    ]
    g = GridGraph(grid)
    comps = g.query("components")
    assert len(comps) == 2
    dist = g.query("bfs", start=(2, 0))
    assert dist[2][2] == 2
    g.update(1, 0, ".")  # 左列打通上下
    assert len(g.query("components")) == 1
    g.delete(1, 0)       # 再封上
    assert len(g.query("components")) == 2
    print("Grid OK", len(comps), dist[2][2])


APPLICATION = """
- 网格 = 隐式图；多源 BFS 把所有源入队即可。
- 岛屿/感染/最短路三件套；障碍用 '*'。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
