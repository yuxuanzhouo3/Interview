"""
7. Find Median from Data Stream
Template: query(median) / update(=add) / add / del(lazy/hard)
App: LC 295。
"""

import heapq


class MedianFinder:
    def __init__(self):
        self.lo = []  # max-heap via negative
        self.hi = []  # min-heap

    def add(self, num):
        self.addNum(num)

    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def update(self, num):
        self.addNum(num)
        return True

    def query(self):
        return self.findMedian()

    def findMedian(self):
        if not self.lo:
            return None
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2.0

    def delete(self, num):
        """教学：重建（生产用懒删除计数）。"""
        vals = [-x for x in self.lo] + list(self.hi)
        if num not in vals:
            return False
        vals.remove(num)
        self.lo, self.hi = [], []
        for v in vals:
            self.addNum(v)
        return True


def _test():
    m = MedianFinder()
    m.addNum(1)
    m.addNum(2)
    assert m.findMedian() == 1.5
    m.addNum(3)
    assert m.findMedian() == 2.0
    print("MedianFinder OK")


APPLICATION = """
- 小半 max-heap、大半 min-heap；平衡长度差 ≤1。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
