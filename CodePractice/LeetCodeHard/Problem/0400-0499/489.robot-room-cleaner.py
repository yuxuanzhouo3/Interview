"""
LC 489. Robot Room Cleaner — Hard (Premium)
https://leetcode.com/problems/robot-room-cleaner/
Best: DFS backtracking with relative coords — O(cells)
"""

from typing import List


# Mock Robot interface for local testing
class Robot:
    def __init__(self, room: List[List[int]], row: int, col: int):
        self.room = room
        self.r = row
        self.c = col
        self.dir = 0  # 0 up, 1 right, 2 down, 3 left
        self.dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.cleaned = set()

    def move(self) -> bool:
        nr = self.r + self.dirs[self.dir][0]
        nc = self.c + self.dirs[self.dir][1]
        m, n = len(self.room), len(self.room[0])
        if 0 <= nr < m and 0 <= nc < n and self.room[nr][nc] == 1:
            self.r, self.c = nr, nc
            return True
        return False

    def turnLeft(self) -> None:
        self.dir = (self.dir - 1) % 4

    def turnRight(self) -> None:
        self.dir = (self.dir + 1) % 4

    def clean(self) -> None:
        self.cleaned.add((self.r, self.c))


class Solution:
    def cleanRoom(self, robot: "Robot") -> None:
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        visited = set()

        def go_back():
            robot.turnRight()
            robot.turnRight()
            robot.move()
            robot.turnRight()
            robot.turnRight()

        def dfs(x: int, y: int, d: int):
            visited.add((x, y))
            robot.clean()
            for i in range(4):
                nd = (d + i) % 4
                nx, ny = x + dirs[nd][0], y + dirs[nd][1]
                if (nx, ny) not in visited and robot.move():
                    dfs(nx, ny, nd)
                    go_back()
                robot.turnRight()

        dfs(0, 0, 0)


def _test():
    room = [
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
    robot = Robot(room, 1, 3)
    Solution().cleanRoom(robot)
    open_cells = {
        (i, j)
        for i in range(len(room))
        for j in range(len(room[0]))
        if room[i][j] == 1
    }
    assert robot.cleaned == open_cells
    # small room
    room2 = [[1, 1], [1, 1]]
    robot2 = Robot(room2, 0, 0)
    Solution().cleanRoom(robot2)
    assert len(robot2.cleaned) == 4
    print("LC 489 OK")


if __name__ == "__main__":
    _test()
