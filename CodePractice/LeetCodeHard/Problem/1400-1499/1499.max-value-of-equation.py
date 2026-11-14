"""
LC 1499. Max Value of Equation — Hard
https://leetcode.com/problems/max-value-of-equation/
Best: Mono deque on yi-xi — O(n)
"""

from typing import List
from collections import deque


class Solution:
    def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
        # yi + yj + |xi - xj| = yi + yj + xj - xi (xi<=xj)
        # = (yi - xi) + (yj + xj)
        dq = deque()  # (xi, yi-xi) decreasing yi-xi
        ans = float("-inf")
        for x, y in points:
            while dq and x - dq[0][0] > k:
                dq.popleft()
            if dq:
                ans = max(ans, y + x + dq[0][1])
            v = y - x
            while dq and dq[-1][1] <= v:
                dq.pop()
            dq.append((x, v))
        return ans


def _test():
    s = Solution()
    assert s.findMaxValueOfEquation([[1, 3], [2, 0], [5, 10], [6, -10]], 1) == 4
    assert s.findMaxValueOfEquation([[0, 0], [3, 0], [9, 2]], 3) == 3
    print("LC 1499 OK")


if __name__ == "__main__":
    _test()
