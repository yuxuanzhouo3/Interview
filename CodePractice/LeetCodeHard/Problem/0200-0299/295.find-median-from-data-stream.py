"""
LC 295. Find Median from Data Stream — Hard
Best: Two heaps (max-left / min-right)
"""

import heapq


class MedianFinder:
    def __init__(self):
        self.lo = []  # max-heap via negatives
        self.hi = []  # min-heap

    def addNum(self, num: int) -> None:
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self) -> float:
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2.0


def _test():
    m = MedianFinder()
    m.addNum(1)
    m.addNum(2)
    assert m.findMedian() == 1.5
    m.addNum(3)
    assert m.findMedian() == 2.0
    print("LC 295 OK")


if __name__ == "__main__":
    _test()
