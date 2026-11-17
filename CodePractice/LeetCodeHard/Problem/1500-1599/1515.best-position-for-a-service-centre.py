"""
LC 1515. Best Position for a Service Centre — Hard
https://leetcode.com/problems/best-position-for-a-service-centre/
Best: Geometric median via gradient descent — O(iter · n)
"""

import math
from typing import List


class Solution:
    def getMinDistSum(self, positions: List[List[int]]) -> float:
        n = len(positions)
        x = sum(p[0] for p in positions) / n
        y = sum(p[1] for p in positions) / n
        lr = 1.0
        eps = 1e-8

        def dist_sum(cx, cy):
            return sum(math.hypot(cx - px, cy - py) for px, py in positions)

        for _ in range(10000):
            gx = gy = 0.0
            for px, py in positions:
                dx, dy = x - px, y - py
                d = math.hypot(dx, dy)
                if d > eps:
                    gx += dx / d
                    gy += dy / d
            nx, ny = x - lr * gx, y - lr * gy
            if dist_sum(nx, ny) > dist_sum(x, y):
                lr *= 0.5
                continue
            if abs(nx - x) + abs(ny - y) < eps:
                x, y = nx, ny
                break
            x, y = nx, ny
        return dist_sum(x, y)


def _test():
    s = Solution()
    assert abs(s.getMinDistSum([[0, 1], [1, 0], [1, 2], [2, 1]]) - 4.0) < 1e-5
    assert abs(s.getMinDistSum([[1, 1], [3, 3]]) - 2.828427) < 1e-5
    print("LC 1515 OK")


if __name__ == "__main__":
    _test()
