"""
LC 3369. Design an Array Statistics Tracker — Hard
https://leetcode.com/problems/design-an-array-statistics-tracker/
Best: Sorted list via bisect + lazy mode heap — O(log n)
"""

from collections import defaultdict, deque
from bisect import bisect_left, insort
import heapq


class StatisticsTracker:
    def __init__(self):
        self.q = deque()
        self.sl = []
        self.freq = defaultdict(int)
        self.freq_heap = []
        self.total = 0

    def addNumber(self, num: int) -> None:
        self.q.append(num)
        insort(self.sl, num)
        self.freq[num] += 1
        self.total += num
        heapq.heappush(self.freq_heap, (-self.freq[num], num))

    def removeFirst(self) -> None:
        if not self.q:
            return
        num = self.q.popleft()
        i = bisect_left(self.sl, num)
        self.sl.pop(i)
        self.freq[num] -= 1
        self.total -= num
        heapq.heappush(self.freq_heap, (-self.freq[num], num))

    def getMean(self) -> int:
        if not self.q:
            return -1
        return self.total // len(self.q)

    def getMedian(self) -> int:
        if not self.q:
            return -1
        return self.sl[len(self.sl) // 2]

    def getMode(self) -> int:
        if not self.q:
            return -1
        while self.freq_heap:
            f, v = self.freq_heap[0]
            if -f == self.freq[v] and self.freq[v] > 0:
                return v
            heapq.heappop(self.freq_heap)
        return -1


def _test():
    t = StatisticsTracker()
    t.addNumber(1)
    t.addNumber(2)
    t.addNumber(2)
    assert t.getMode() == 2
    assert t.getMedian() == 2
    assert t.getMean() == 1
    t.removeFirst()
    assert t.getMean() == 2
    print("LC 3369 OK")


if __name__ == "__main__":
    _test()
