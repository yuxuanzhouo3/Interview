"""
LC 1776. Car Fleet II — Hard
https://leetcode.com/problems/car-fleet-ii/
Best: Monotonic stack (right to left) — O(n)
"""

from typing import List


class Solution:
    def getCollisionTimes(self, cars: List[List[int]]) -> List[float]:
        n = len(cars)
        ans = [-1.0] * n
        stack = []  # indices of candidates ahead
        for i in range(n - 1, -1, -1):
            p, s = cars[i]
            while stack:
                j = stack[-1]
                pj, sj = cars[j]
                if s <= sj:
                    stack.pop()
                    continue
                t = (pj - p) / (s - sj)
                if ans[j] < 0 or t <= ans[j]:
                    ans[i] = t
                    break
                stack.pop()
            stack.append(i)
        return ans


def _test():
    s = Solution()
    r = s.getCollisionTimes([[1, 2], [2, 1], [4, 3], [7, 2]])
    assert abs(r[0] - 1.0) < 1e-5 and r[1] == -1.0 and abs(r[2] - 3.0) < 1e-5 and r[3] == -1.0
    r2 = s.getCollisionTimes([[3, 4], [5, 4], [6, 3], [9, 1]])
    assert abs(r2[0] - 2.0) < 1e-5 and abs(r2[1] - 1.0) < 1e-5 and abs(r2[2] - 1.5) < 1e-5 and r2[3] == -1.0
    print("LC 1776 OK")


if __name__ == "__main__":
    _test()
