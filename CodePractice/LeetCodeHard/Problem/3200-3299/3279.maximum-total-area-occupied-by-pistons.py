"""
LC 3279. Maximum Total Area Occupied by Pistons — Hard
https://leetcode.com/problems/maximum-total-area-occupied-by-pistons/
Best: Evaluate at bounce critical times — O(n^2)
"""

from typing import List


class Solution:
    def maxArea(self, height: int, positions: List[int], directions: str) -> int:
        n = len(positions)
        h = height
        period = 2 * h

        def pos_at(i: int, t: int) -> int:
            p0 = positions[i]
            if directions[i] == "U":
                x = (p0 + t) % period
            else:
                x = (p0 - t) % period
            if x < 0:
                x += period
            if x <= h:
                return x
            return 2 * h - x

        crit = {0}
        for i, p in enumerate(positions):
            if directions[i] == "U":
                crit.add(h - p)
                crit.add(h - p + h)
            else:
                crit.add(p)
                crit.add(p + h)
            crit.add(period)

        best = 0
        for t in crit:
            if 0 <= t <= period:
                best = max(best, sum(pos_at(i, t) for i in range(n)))
        return best


def _test():
    s = Solution()
    assert s.maxArea(5, [2, 4], "UD") >= 6
    assert s.maxArea(3, [1, 2], "UU") >= 3
    print("LC 3279 OK")


if __name__ == "__main__":
    _test()
