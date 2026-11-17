"""
LC 716. Max Stack — Hard
https://leetcode.com/problems/max-stack/
Best: Stack + max-heap with lazy deletion — O(log n)
"""

import heapq


class MaxStack:
    def __init__(self):
        self.stack = []  # (val, id)
        self.heap = []  # (-val, -id)
        self.removed = set()
        self.uid = 0

    def _clean_stack(self):
        while self.stack and self.stack[-1][1] in self.removed:
            self.removed.remove(self.stack[-1][1])
            self.stack.pop()

    def _clean_heap(self):
        while self.heap and -self.heap[0][1] in self.removed:
            self.removed.remove(-self.heap[0][1])
            heapq.heappop(self.heap)

    def push(self, x: int) -> None:
        self.uid += 1
        self.stack.append((x, self.uid))
        heapq.heappush(self.heap, (-x, -self.uid))

    def pop(self) -> int:
        self._clean_stack()
        val, i = self.stack.pop()
        self.removed.add(i)
        return val

    def top(self) -> int:
        self._clean_stack()
        return self.stack[-1][0]

    def peekMax(self) -> int:
        self._clean_heap()
        return -self.heap[0][0]

    def popMax(self) -> int:
        self._clean_heap()
        val, i = -self.heap[0][0], -self.heap[0][1]
        heapq.heappop(self.heap)
        self.removed.add(i)
        return val


def _test():
    stk = MaxStack()
    stk.push(5)
    stk.push(1)
    stk.push(5)
    assert stk.top() == 5
    assert stk.popMax() == 5
    assert stk.top() == 1
    assert stk.peekMax() == 5
    assert stk.pop() == 1
    assert stk.top() == 5
    print("LC 716 OK")


if __name__ == "__main__":
    _test()
