"""
LC 302. Smallest Rectangle Enclosing Black Pixels — Hard
https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/
Best: 四次二分找边界 — O(m log n + n log m)
"""

from typing import List


class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        m, n = len(image), len(image[0])

        def col_has_black(c: int, lo: int, hi: int) -> bool:
            for r in range(lo, hi):
                if image[r][c] == "1":
                    return True
            return False

        def row_has_black(r: int, lo: int, hi: int) -> bool:
            for c in range(lo, hi):
                if image[r][c] == "1":
                    return True
            return False

        # left: 最左含黑列 in [0, y]
        lo, hi = 0, y
        while lo < hi:
            mid = (lo + hi) // 2
            if col_has_black(mid, 0, m):
                hi = mid
            else:
                lo = mid + 1
        left = lo

        # right: 最右含黑列+1 in [y, n)
        lo, hi = y, n
        while lo < hi:
            mid = (lo + hi) // 2
            if col_has_black(mid, 0, m):
                lo = mid + 1
            else:
                hi = mid
        right = lo

        # top
        lo, hi = 0, x
        while lo < hi:
            mid = (lo + hi) // 2
            if row_has_black(mid, left, right):
                hi = mid
            else:
                lo = mid + 1
        top = lo

        # bottom
        lo, hi = x, m
        while lo < hi:
            mid = (lo + hi) // 2
            if row_has_black(mid, left, right):
                lo = mid + 1
            else:
                hi = mid
        bottom = lo

        return (right - left) * (bottom - top)


def _test():
    s = Solution()
    img = [["0", "0", "1", "0"], ["0", "1", "1", "0"], ["0", "1", "0", "0"]]
    assert s.minArea(img, 0, 2) == 6
    img2 = [["1"]]
    assert s.minArea(img2, 0, 0) == 1
    print("LC 302 OK")


if __name__ == "__main__":
    _test()
