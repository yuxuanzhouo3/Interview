"""
LC 1825. Finding MK Average — Hard
https://leetcode.com/problems/finding-mk-average/
Best: Two heaps + multiset mid sum — O(log m) per op
"""


from sortedcontainers import SortedList
from collections import deque


class MKAverage:
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.q = deque()
        self.sorted = SortedList()
        self.total = 0

    def addElement(self, num: int) -> None:
        self.q.append(num)
        self.sorted.add(num)
        self.total += num
        if len(self.q) > self.m:
            old = self.q.popleft()
            self.sorted.remove(old)
            self.total -= old

    def calculateMKAverage(self) -> int:
        if len(self.q) < self.m:
            return -1
        k = self.k
        s = self.total
        for i in range(k):
            s -= self.sorted[i]
            s -= self.sorted[-(i + 1)]
        return s // (self.m - 2 * k)


def _test():
    obj = MKAverage(3, 1)
    obj.addElement(3)
    obj.addElement(1)
    assert obj.calculateMKAverage() == -1
    obj.addElement(10)
    assert obj.calculateMKAverage() == 3
    obj.addElement(5)
    obj.addElement(5)
    obj.addElement(5)
    assert obj.calculateMKAverage() == 5
    print("LC 1825 OK")


if __name__ == "__main__":
    _test()

