"""
LC 1172. Dinner Plate Stacks — Hard
https://leetcode.com/problems/dinner-plate-stacks/
Best: Stacks + min-heap of non-full indices — O(log n) per op
"""

import heapq


class DinnerPlates:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.stacks = []
        self.avail = []  # min-heap of non-full stack indices

    def push(self, val: int) -> None:
        while self.avail and (
            self.avail[0] >= len(self.stacks) or len(self.stacks[self.avail[0]]) == self.cap
        ):
            heapq.heappop(self.avail)
        if not self.avail:
            self.stacks.append([])
            heapq.heappush(self.avail, len(self.stacks) - 1)
        i = self.avail[0]
        self.stacks[i].append(val)
        if len(self.stacks[i]) == self.cap:
            heapq.heappop(self.avail)

    def pop(self) -> int:
        while self.stacks and not self.stacks[-1]:
            self.stacks.pop()
        if not self.stacks:
            return -1
        return self.popAtStack(len(self.stacks) - 1)

    def popAtStack(self, index: int) -> int:
        if index < 0 or index >= len(self.stacks) or not self.stacks[index]:
            return -1
        if len(self.stacks[index]) == self.cap:
            heapq.heappush(self.avail, index)
        return self.stacks[index].pop()


def _test():
    d = DinnerPlates(2)
    d.push(1)
    d.push(2)
    d.push(3)
    d.push(4)
    d.push(5)
    assert d.popAtStack(0) == 2
    d.push(20)
    d.push(21)
    assert d.popAtStack(0) == 20
    assert d.popAtStack(2) == 21
    assert d.pop() == 5
    assert d.pop() == 4
    assert d.pop() == 3
    assert d.pop() == 1
    assert d.pop() == -1
    print("LC 1172 OK")


if __name__ == "__main__":
    _test()
