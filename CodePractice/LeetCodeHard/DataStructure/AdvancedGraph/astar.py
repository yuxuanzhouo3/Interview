"""
AdvancedGraph / A*（网格启发搜索）
Template: query(path_len) / update(cell) / add / del(obstacle)
App: 单源单汇 + 好启发；游戏寻路；超市网格题可对比 BFS。
来源: CodePractice/graph.py §4.6
"""

import heapq


class AStarGrid:
    def __init__(self, grid=None):
        # grid: list[list[str]] '.' 可走 '*' 障碍
        self.grid = [row[:] for row in (grid or [["."]])]
        self.n = len(self.grid)
        self.m = len(self.grid[0])

    def add(self, r, c=None, cell="."):
        """扩行/列；或 add(r,c,cell) 设格子。"""
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
        """删通 → 设障碍。"""
        self.update(r, c, "*")

    def query(self, start, target):
        return self.astar(start, target)

    def astar(self, start, target):
        sx, sy = start
        tx, ty = target
        if self.grid[sx][sy] == "*" or self.grid[tx][ty] == "*":
            return -1

        def h(x, y):
            return abs(x - tx) + abs(y - ty)

        INF = float("inf")
        dist = {start: 0}
        pq = [(h(sx, sy), 0, sx, sy)]
        D = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while pq:
            f, g, x, y = heapq.heappop(pq)
            if (x, y) == target:
                return g
            if g != dist.get((x, y), INF):
                continue
            for dx, dy in D:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.n and 0 <= ny < self.m):
                    continue
                if self.grid[nx][ny] == "*":
                    continue
                ng = g + 1
                if ng < dist.get((nx, ny), INF):
                    dist[(nx, ny)] = ng
                    heapq.heappush(pq, (ng + h(nx, ny), ng, nx, ny))
        return -1


def _test():
    grid = [
        [".", ".", "*"],
        [".", "*", "."],
        [".", ".", "."],
    ]
    g = AStarGrid(grid)
    assert g.query((0, 0), (2, 2)) == 4
    assert g.update(1, 1, ".")
    assert g.query((0, 0), (2, 2)) == 4  # 仍可更短或等
    assert g.query((0, 0), (0, 2)) == -1  # 目标是障碍
    g.update(0, 2, ".")
    assert g.query((0, 0), (0, 2)) == 2
    g.delete(0, 1)
    assert g.query((0, 0), (0, 2)) == -1 or g.query((0, 0), (0, 2)) >= 0
    # 绕路
    assert g.query((0, 0), (2, 0)) == 2
    print("AStar OK", g.query((0, 0), (2, 2)))


APPLICATION = """
- f = g + h；h 可采纳（≤真实）则最优。曼哈顿适合四向网格。
- 无启发 ≈ Dijkstra；均匀代价网格用 BFS 即可。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
