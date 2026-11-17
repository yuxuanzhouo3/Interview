"""
LC 3235. Check if the Rectangle Corner Is Reachable — Hard
https://leetcode.com/problems/check-if-the-rectangle-corner-is-reachable/
Best: Union-Find circles + borders — O(n^2)
"""

from typing import List


class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # start (0,0) to (xCorner,yCorner), axis-aligned rect, cannot touch/enter circles
        # blocked if circles connect left-top or bottom-right in a cutting way
        # UF: circles touching each other; also touch borders
        n = len(circles)
        parent = list(range(n + 2))
        LEFT_TOP, BOTTOM_RIGHT = n, n + 1

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        def touches(i, j):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dx, dy = x1 - x2, y1 - y2
            return dx * dx + dy * dy <= (r1 + r2) * (r1 + r2)

        for i, (x, y, r) in enumerate(circles):
            # touch left or top border of rectangle [0,xCorner]x[0,yCorner]
            if x - r <= 0 or y + r >= yCorner:
                union(i, LEFT_TOP)
            # touch bottom or right
            if y - r <= 0 or x + r >= xCorner:
                union(i, BOTTOM_RIGHT)
            # also block corners if circle covers start/end
            if x * x + y * y <= r * r:
                return False
            if (x - xCorner) ** 2 + (y - yCorner) ** 2 <= r * r:
                return False

        for i in range(n):
            for j in range(i + 1, n):
                if touches(i, j):
                    union(i, j)

        return find(LEFT_TOP) != find(BOTTOM_RIGHT)


def _test():
    s = Solution()
    assert s.canReachCorner(3, 4, [[2, 1, 1]]) is True
    assert s.canReachCorner(3, 3, [[1, 1, 2]]) is False
    assert s.canReachCorner(3, 4, [[2, 1, 1], [1, 2, 1]]) is False
    print("LC 3235 OK")


if __name__ == "__main__":
    _test()
