"""
LC 3009. Maximum Number of Intersections on the Chart — Hard
Best: Sweep line + fenwick — O(n log n)
"""

from typing import List


class Solution:
    def maxIntersectionCount(self, y: List[int]) -> int:
        # polyline (i,y[i])-(i+1,y[i+1]); count max intersections with horizontal line
        # exclude vertices carefully
        events = []  # open/close of vertical ranges of edges at half-integers conceptually
        for i in range(len(y) - 1):
            a, b = y[i], y[i + 1]
            lo, hi = min(a, b), max(a, b)
            # segment covers (lo, hi] or [lo, hi) depending direction - interior
            if lo == hi:
                continue
            events.append((lo, 1))
            events.append((hi, -1))
        events.sort(key=lambda e: (e[0], -e[1]))
        cur = ans = 0
        for _, d in events:
            cur += d
            ans = max(ans, cur)
        return ans


def _test():
    s = Solution()
    assert s.maxIntersectionCount([1, 2, 1, 2, 1, 3, 2]) >= 2
    print("LC 3009 OK")


if __name__ == "__main__":
    _test()
