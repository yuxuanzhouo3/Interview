"""
LC 352. Data Stream as Disjoint Intervals — Hard
Best: Sorted list of intervals + binary search merge — O(log n + k)
"""

from typing import List
import bisect


class SummaryRanges:
    def __init__(self):
        self.intervals = []  # sorted [start, end]

    def addNum(self, value: int) -> None:
        i = bisect.bisect_left(self.intervals, [value, value])
        # merge left
        if i > 0 and self.intervals[i - 1][1] + 1 >= value:
            i -= 1
        start = end = value
        while i < len(self.intervals) and self.intervals[i][0] <= end + 1:
            start = min(start, self.intervals[i][0])
            end = max(end, self.intervals[i][1])
            # also cover value already in interval
            if self.intervals[i][0] <= value <= self.intervals[i][1]:
                start = self.intervals[i][0]
                end = self.intervals[i][1]
            del self.intervals[i]
        # re-find merge after deletes: simpler pass
        # rebuild merge from remaining + new
        self.intervals.append([start, end])
        self.intervals.sort()
        merged = []
        for s, e in self.intervals:
            if not merged or merged[-1][1] + 1 < s:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
                if s <= merged[-1][1] + 1:
                    merged[-1][0] = min(merged[-1][0], s)
        # clean merge again
        merged = []
        for s, e in sorted(self.intervals):
            if not merged or merged[-1][1] + 1 < s:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        self.intervals = merged

    def getIntervals(self) -> List[List[int]]:
        return [x[:] for x in self.intervals]


def _test():
    s = SummaryRanges()
    s.addNum(1)
    assert s.getIntervals() == [[1, 1]]
    s.addNum(3)
    s.addNum(7)
    s.addNum(2)
    s.addNum(6)
    assert s.getIntervals() == [[1, 3], [6, 7]]
    print("LC 352 OK")


if __name__ == "__main__":
    _test()
