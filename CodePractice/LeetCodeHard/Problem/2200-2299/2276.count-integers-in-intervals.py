"""
LC 2276. Count Integers in Intervals — Hard
https://leetcode.com/problems/count-integers-in-intervals/
Best: SortedList merge intervals track coverage — O(log n)
"""


from sortedcontainers import SortedList


class CountIntervals:
    def __init__(self):
        self.intervals = SortedList()  # (left, right)
        self.covered = 0

    def add(self, left: int, right: int) -> None:
        i = self.intervals.bisect_left((left, right))
        # merge with previous overlapping
        if i > 0 and self.intervals[i - 1][1] >= left - 1:
            i -= 1
        new_l, new_r = left, right
        while i < len(self.intervals) and self.intervals[i][0] <= new_r + 1:
            l, r = self.intervals.pop(i)
            self.covered -= r - l + 1
            new_l = min(new_l, l)
            new_r = max(new_r, r)
        self.intervals.add((new_l, new_r))
        self.covered += new_r - new_l + 1

    def count(self) -> int:
        return self.covered


def _test():
    c = CountIntervals()
    c.add(2, 3)
    c.add(7, 10)
    assert c.count() == 6
    c.add(5, 8)
    assert c.count() == 8
    print("LC 2276 OK")


if __name__ == "__main__":
    _test()

