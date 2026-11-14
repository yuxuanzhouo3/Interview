"""
LC 732. My Calendar III — Hard
Best: Diff map sweep — O(n log n) per book
"""


class MyCalendarThree:
    def __init__(self):
        self.diff = {}

    def book(self, startTime: int, endTime: int) -> int:
        self.diff[startTime] = self.diff.get(startTime, 0) + 1
        self.diff[endTime] = self.diff.get(endTime, 0) - 1
        cur = best = 0
        for t in sorted(self.diff):
            cur += self.diff[t]
            best = max(best, cur)
        return best


def _test():
    c = MyCalendarThree()
    assert c.book(10, 20) == 1
    assert c.book(50, 60) == 1
    assert c.book(10, 40) == 2
    assert c.book(5, 15) == 3
    print("LC 732 OK")


if __name__ == "__main__":
    _test()
