"""
LC 149. Max Points on a Line — Hard
Best: For each origin, slope counter — O(n^2)
"""

from typing import List
from collections import defaultdict
from math import gcd


class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        if n <= 2:
            return n
        ans = 1
        for i in range(n):
            slopes = defaultdict(int)
            same = 1
            local = 0
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    same += 1
                    continue
                g = gcd(dx, dy)
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                slopes[(dx, dy)] += 1
                local = max(local, slopes[(dx, dy)])
            ans = max(ans, local + same)
        return ans


def _test():
    s = Solution()
    assert s.maxPoints([[1, 1], [2, 2], [3, 3]]) == 3
    assert s.maxPoints([[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]]) == 4
    print("LC 149 OK")


if __name__ == "__main__":
    _test()
