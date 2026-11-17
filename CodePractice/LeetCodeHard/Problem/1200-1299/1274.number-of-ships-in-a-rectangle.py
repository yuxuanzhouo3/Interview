"""
LC 1274. Number of Ships in a Rectangle — Hard
https://leetcode.com/problems/number-of-ships-in-a-rectangle/
Best: 四叉分割 + hasShips 剪枝 — O(ships * log(area))
"""


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Sea:
    def __init__(self, ships):
        self.ships = {(x, y) for x, y in ships}
        self.calls = 0

    def hasShips(self, topRight: "Point", bottomLeft: "Point") -> bool:
        self.calls += 1
        return any(
            bottomLeft.x <= x <= topRight.x and bottomLeft.y <= y <= topRight.y
            for x, y in self.ships
        )


class Solution:
    def countShips(self, sea: "Sea", topRight: "Point", bottomLeft: "Point") -> int:
        if topRight.x < bottomLeft.x or topRight.y < bottomLeft.y:
            return 0
        if not sea.hasShips(topRight, bottomLeft):
            return 0
        if topRight.x == bottomLeft.x and topRight.y == bottomLeft.y:
            return 1
        mx = (topRight.x + bottomLeft.x) // 2
        my = (topRight.y + bottomLeft.y) // 2
        return (
            self.countShips(sea, Point(mx, my), bottomLeft)
            + self.countShips(sea, Point(topRight.x, my), Point(mx + 1, bottomLeft.y))
            + self.countShips(sea, Point(mx, topRight.y), Point(bottomLeft.x, my + 1))
            + self.countShips(sea, topRight, Point(mx + 1, my + 1))
        )


def _test():
    sea = Sea([[1, 1], [2, 2], [3, 3]])
    s = Solution()
    assert s.countShips(sea, Point(4, 4), Point(0, 0)) == 3
    sea2 = Sea([[1, 1], [2, 2], [3, 3]])
    assert s.countShips(sea2, Point(1000, 1000), Point(0, 0)) == 3
    print("LC 1274 OK")


if __name__ == "__main__":
    _test()
