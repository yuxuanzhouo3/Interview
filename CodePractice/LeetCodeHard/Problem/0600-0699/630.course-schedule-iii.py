"""
LC 630. Course Schedule III — Hard
https://leetcode.com/problems/course-schedule-iii/
Best: sort by deadline + max-heap of durations — O(n log n)
"""

import heapq
from typing import List


class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key=lambda x: x[1])
        heap = []  # max-heap via negation
        time = 0
        for dur, last in courses:
            if time + dur <= last:
                heapq.heappush(heap, -dur)
                time += dur
            elif heap and -heap[0] > dur:
                time += dur + heapq.heappop(heap)
                heapq.heappush(heap, -dur)
        return len(heap)


def _test():
    s = Solution()
    assert s.scheduleCourse([[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]) == 3
    assert s.scheduleCourse([[1, 2]]) == 1
    assert s.scheduleCourse([[3, 2], [4, 3]]) == 0
    print("LC 630 OK")


if __name__ == "__main__":
    _test()
