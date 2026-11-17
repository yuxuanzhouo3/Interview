"""
LC 1610. Maximum Number of Visible Points — Hard
https://leetcode.com/problems/maximum-number-of-visible-points/
Best: atan2 angles + duplicate+2π sliding window — O(n log n)
"""

from typing import List
import math


class Solution:
    def visiblePoints(self, points: List[List[int]], angle: int, location: List[int]) -> int:
        lx, ly = location
        same = 0
        angs = []
        for x, y in points:
            if x == lx and y == ly:
                same += 1
            else:
                angs.append(math.atan2(y - ly, x - lx))
        angs.sort()
        n = len(angs)
        angs += [a + 2 * math.pi for a in angs]
        rad = math.radians(angle)
        best = 0
        j = 0
        for i in range(n):
            while j < len(angs) and angs[j] - angs[i] <= rad + 1e-9:
                j += 1
            best = max(best, j - i)
        return best + same


def _test():
    s = Solution()
    assert s.visiblePoints([[2, 1], [2, 2], [3, 3]], 90, [1, 1]) == 3
    assert s.visiblePoints([[2, 1], [2, 2], [3, 4], [1, 1]], 90, [1, 1]) == 4
    assert s.visiblePoints([[1, 0], [2, 1]], 13, [1, 1]) == 1
    print("LC 1610 OK")


if __name__ == "__main__":
    _test()
