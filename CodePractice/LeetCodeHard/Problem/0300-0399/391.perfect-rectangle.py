"""
LC 391. Perfect Rectangle — Hard
Best: Area + corner point parity — O(n)
"""

from typing import List


class Solution:
    def isRectangleCover(self, rectangles: List[List[int]]) -> bool:
        area = 0
        corners = set()
        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        for x1, y1, x2, y2 in rectangles:
            area += (x2 - x1) * (y2 - y1)
            minx, miny = min(minx, x1), min(miny, y1)
            maxx, maxy = max(maxx, x2), max(maxy, y2)
            for p in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
                if p in corners:
                    corners.remove(p)
                else:
                    corners.add(p)
        if corners != {(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)}:
            return False
        return area == (maxx - minx) * (maxy - miny)


def _test():
    s = Solution()
    assert s.isRectangleCover([[1, 1, 3, 3], [3, 1, 4, 2], [3, 2, 4, 4], [1, 3, 2, 4], [2, 3, 3, 4]]) is True
    assert s.isRectangleCover([[1, 1, 2, 3], [1, 3, 2, 4], [3, 1, 4, 2], [3, 2, 4, 4]]) is False
    print("LC 391 OK")


if __name__ == "__main__":
    _test()
