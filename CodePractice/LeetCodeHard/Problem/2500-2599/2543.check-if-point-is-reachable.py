"""
LC 2543. Check if Point is Reachable — Hard
https://leetcode.com/problems/check-if-point-is-reachable/
Best: Backward / gcd only factor 2 — O(log)
"""

import math


class Solution:
    def isReachable(self, targetX: int, targetY: int) -> bool:
        g = math.gcd(targetX, targetY)
        return g > 0 and (g & (g - 1)) == 0


def _test():
    s = Solution()
    assert s.isReachable(6, 9) is False
    assert s.isReachable(4, 7) is True
    assert s.isReachable(1, 1) is True
    print("LC 2543 OK")


if __name__ == "__main__":
    _test()
