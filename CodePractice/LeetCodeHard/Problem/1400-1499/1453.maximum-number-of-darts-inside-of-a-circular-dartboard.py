"""
LC 1453. Maximum Number of Darts Inside of a Circular Dartboard — Hard
https://leetcode.com/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/
Best: Enumerate circle through 2 points — O(n^3)
"""

import math
from typing import List


class Solution:
    def numPoints(self, darts: List[List[int]], r: int) -> int:
        n = len(darts)
        if n == 1:
            return 1
        ans = 1
        r2 = r * r

        def centers(i, j):
            x1, y1 = darts[i]
            x2, y2 = darts[j]
            dx, dy = x2 - x1, y2 - y1
            d2 = dx * dx + dy * dy
            if d2 > 4 * r2:
                return []
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if d2 == 0:
                return [(x1, y1)]
            h = math.sqrt(r2 - d2 / 4)
            ux, uy = -dy / math.sqrt(d2), dx / math.sqrt(d2)
            return [(mx + ux * h, my + uy * h), (mx - ux * h, my - uy * h)]

        def count(cx, cy):
            c = 0
            for x, y in darts:
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2 + 1e-8:
                    c += 1
            return c

        for i in range(n):
            for j in range(i + 1, n):
                for cx, cy in centers(i, j):
                    ans = max(ans, count(cx, cy))
        return ans


def _test():
    s = Solution()
    assert s.numPoints([[-2, 0], [2, 0], [0, 2], [0, -2]], 2) == 4
    assert s.numPoints([[-3, 0], [3, 0], [2, 6], [5, 4], [0, 9], [7, 8]], 5) == 5
    print("LC 1453 OK")


if __name__ == "__main__":
    _test()
