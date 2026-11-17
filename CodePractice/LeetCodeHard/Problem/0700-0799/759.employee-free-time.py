"""
LC 759. Employee Free Time — Hard
https://leetcode.com/problems/employee-free-time/
Tags: Array, Sorting, Heap
Best: 合并所有忙碌区间后取空隙
"""

from typing import List, Optional


class Interval:
    def __init__(self, start: int = 0, end: int = 0):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __repr__(self):
        return f"[{self.start},{self.end}]"


class Solution:
    def employeeFreeTime(self, schedule: List[List[Interval]]) -> List[Interval]:
        intervals = sorted((iv for emp in schedule for iv in emp), key=lambda x: x.start)
        if not intervals:
            return []
        merged = []
        cur_s, cur_e = intervals[0].start, intervals[0].end
        for iv in intervals[1:]:
            if iv.start <= cur_e:
                cur_e = max(cur_e, iv.end)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = iv.start, iv.end
        merged.append((cur_s, cur_e))
        ans = []
        for i in range(1, len(merged)):
            ans.append(Interval(merged[i - 1][1], merged[i][0]))
        return ans


def _test():
    sol = Solution()
    sch = [[Interval(1, 2), Interval(5, 6)], [Interval(1, 3)], [Interval(4, 10)]]
    assert sol.employeeFreeTime(sch) == [Interval(3, 4)]
    sch2 = [[Interval(1, 3), Interval(6, 7)], [Interval(2, 4)], [Interval(2, 5), Interval(9, 12)]]
    assert sol.employeeFreeTime(sch2) == [Interval(5, 6), Interval(7, 9)]
    print("LC 759 OK")


if __name__ == "__main__":
    _test()

