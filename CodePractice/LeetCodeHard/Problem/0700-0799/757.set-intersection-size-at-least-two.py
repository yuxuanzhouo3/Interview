"""
LC 757. Set Intersection Size At Least Two — Hard
Best: Sort by end, greedy place two points — O(n log n)
"""

from typing import List


class Solution:
    def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: (x[1], -x[0]))
        # keep last two chosen points
        p1 = p2 = -1
        ans = 0
        for a, b in intervals:
            if a <= p1:
                continue
            if a <= p2:
                # need one more point at b
                p1, p2 = p2, b
                ans += 1
            else:
                # need two points b-1, b
                p1, p2 = b - 1, b
                ans += 2
        return ans


def _test():
    s = Solution()
    assert s.intersectionSizeTwo([[1, 3], [1, 4], [2, 5], [3, 5]]) == 3
    assert s.intersectionSizeTwo([[1, 2], [2, 3], [2, 4], [4, 5]]) == 5
    print("LC 757 OK")


if __name__ == "__main__":
    _test()
