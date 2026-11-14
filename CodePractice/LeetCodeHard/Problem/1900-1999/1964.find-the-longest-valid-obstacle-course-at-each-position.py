"""
LC 1964. Find the Longest Valid Obstacle Course at Each Position — Hard
https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/
Best: LIS with bisect_right — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        tails = []
        ans = []
        for x in obstacles:
            j = bisect.bisect_right(tails, x)
            if j == len(tails):
                tails.append(x)
            else:
                tails[j] = x
            ans.append(j + 1)
        return ans


def _test():
    s = Solution()
    assert s.longestObstacleCourseAtEachPosition([1, 2, 3, 2]) == [1, 2, 3, 3]
    assert s.longestObstacleCourseAtEachPosition([2, 2, 1]) == [1, 2, 1]
    assert s.longestObstacleCourseAtEachPosition([3, 1, 5, 6, 4, 2]) == [1, 1, 2, 3, 2, 2]
    print("LC 1964 OK")


if __name__ == "__main__":
    _test()
