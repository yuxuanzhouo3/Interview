"""
LC 335. Self Crossing — Hard
Best: Case analysis on last 4-6 segments — O(n)
"""

from typing import List


class Solution:
    def isSelfCrossing(self, distance: List[int]) -> bool:
        d = distance
        for i in range(3, len(d)):
            if d[i] >= d[i - 2] and d[i - 1] <= d[i - 3]:
                return True
            if i >= 4 and d[i - 1] == d[i - 3] and d[i] + d[i - 4] >= d[i - 2]:
                return True
            if i >= 5 and d[i - 2] >= d[i - 4] and d[i - 1] <= d[i - 3] and d[i - 1] >= d[i - 3] - d[i - 5] and d[i] >= d[i - 2] - d[i - 4]:
                return True
        return False


def _test():
    s = Solution()
    assert s.isSelfCrossing([2, 1, 1, 2]) is True
    assert s.isSelfCrossing([1, 2, 3, 4]) is False
    print("LC 335 OK")


if __name__ == "__main__":
    _test()
