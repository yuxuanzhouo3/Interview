"""
LC 1924. Erect the Fence II — Hard
https://leetcode.com/problems/erect-the-fence-ii/
Best: Welzl 最小覆盖圆 — 期望 O(n)
"""

from typing import List
import random
import math


class Solution:
    def outerTrees(self, trees: List[List[int]]) -> List[float]:
        pts = [tuple(p) for p in trees]
        random.shuffle(pts)
        eps = 1e-9

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def circle2(a, b):
            c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            return c, dist(a, b) / 2

        def circle3(a, b, c):
            ax, ay = a
            bx, by = b[0] - ax, b[1] - ay
            cx, cy = c[0] - ax, c[1] - ay
            b2 = bx * bx + by * by
            c2 = cx * cx + cy * cy
            d = 2 * (bx * cy - by * cx)
            if abs(d) < 1e-12:
                # collinear — take farthest pair
                pairs = [(a, b), (a, c), (b, c)]
                u, v = max(pairs, key=lambda p: dist(p[0], p[1]))
                return circle2(u, v)
            ux = (cy * b2 - by * c2) / d + ax
            uy = (bx * c2 - cx * b2) / d + ay
            center = (ux, uy)
            return center, dist(center, a)

        def trivial(R):
            if not R:
                return (0.0, 0.0), 0.0
            if len(R) == 1:
                return R[0], 0.0
            if len(R) == 2:
                return circle2(R[0], R[1])
            return circle3(R[0], R[1], R[2])

        def inside(cir, p):
            c, r = cir
            return dist(c, p) <= r + eps

        def welzl(i, R):
            if i == len(pts) or len(R) == 3:
                return trivial(R)
            cir = welzl(i + 1, R)
            if inside(cir, pts[i]):
                return cir
            return welzl(i + 1, R + [pts[i]])

        (x, y), r = welzl(0, [])
        return [x, y, r]


def _test():
    s = Solution()
    for trees in [[[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]],
                  [[1, 2], [2, 2], [4, 2]],
                  [[1, 1]]]:
        x, y, r = s.outerTrees(trees)
        for px, py in trees:
            assert (px - x) ** 2 + (py - y) ** 2 <= (r + 1e-5) ** 2 + 1e-6
    print("LC 1924 OK")


if __name__ == "__main__":
    _test()
