"""
LC 3382. Maximum Area Rectangle With Point Constraints II — Hard
https://leetcode.com/problems/maximum-area-rectangle-with-point-constraints-ii/
Best: Sort + Fenwick emptiness check — O(n log n)
"""

from typing import List


class BIT:
    def __init__(self, n: int):
        self.bit = [0] * (n + 1)

    def add(self, i: int, val: int) -> None:
        i += 1
        while i < len(self.bit):
            self.bit[i] += val
            i += i & -i

    def query(self, i: int) -> int:
        i += 1
        ret = 0
        while i > 0:
            ret += self.bit[i]
            i -= i & -i
        return ret


class Solution:
    def maxRectangleArea(self, xCoord: List[int], yCoord: List[int]) -> int:
        points = sorted((xCoord[i], yCoord[i]) for i in range(len(xCoord)))
        y_to_idx = {y: idx for idx, y in enumerate(sorted(set(yCoord)))}
        bit = BIT(len(y_to_idx))
        lookup = {}
        result = -1
        for i, (x, y) in enumerate(points):
            y_idx = y_to_idx[y]
            bit.add(y_idx, 1)
            if not (i - 1 >= 0 and points[i - 1][0] == x):
                continue
            prev_y_idx = y_to_idx[points[i - 1][1]]
            curr = bit.query(y_idx) - bit.query(prev_y_idx - 1)
            key = (prev_y_idx, y_idx)
            if key in lookup and lookup[key][0] == curr - 2:
                result = max(result, (x - lookup[key][1]) * (y - points[i - 1][1]))
            lookup[key] = (curr, x)
        return result


def _test():
    s = Solution()
    assert s.maxRectangleArea([1, 1, 3, 3], [1, 3, 1, 3]) == 4
    assert s.maxRectangleArea([1, 1, 3, 3, 2], [1, 3, 1, 3, 2]) == -1
    assert s.maxRectangleArea([1, 1, 3, 3, 1, 3], [1, 3, 1, 3, 2, 2]) == 2
    print("LC 3382 OK")


if __name__ == "__main__":
    _test()
