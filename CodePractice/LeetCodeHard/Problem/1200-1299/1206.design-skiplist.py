"""
LC 1206. Design Skiplist — Hard
https://leetcode.com/problems/design-skiplist/
Best: Multi-level linked list with random promotion — O(log n) expected
"""

import random
from typing import Optional


class Node:
    __slots__ = ("val", "next")

    def __init__(self, val: int, level: int):
        self.val = val
        self.next = [None] * level


class Skiplist:
    MAX_LEVEL = 16
    P = 0.5

    def __init__(self):
        self.head = Node(-1, self.MAX_LEVEL)
        self.level = 1

    def _rand_level(self) -> int:
        lv = 1
        while lv < self.MAX_LEVEL and random.random() < self.P:
            lv += 1
        return lv

    def _find(self, target: int):
        updates = [None] * self.MAX_LEVEL
        cur = self.head
        for i in range(self.level - 1, -1, -1):
            while cur.next[i] and cur.next[i].val < target:
                cur = cur.next[i]
            updates[i] = cur
        return updates

    def search(self, target: int) -> bool:
        cur = self.head
        for i in range(self.level - 1, -1, -1):
            while cur.next[i] and cur.next[i].val < target:
                cur = cur.next[i]
        cur = cur.next[0]
        return cur is not None and cur.val == target

    def add(self, num: int) -> None:
        updates = self._find(num)
        lv = self._rand_level()
        if lv > self.level:
            for i in range(self.level, lv):
                updates[i] = self.head
            self.level = lv
        node = Node(num, lv)
        for i in range(lv):
            node.next[i] = updates[i].next[i]
            updates[i].next[i] = node

    def erase(self, num: int) -> bool:
        updates = self._find(num)
        node = updates[0].next[0]
        if not node or node.val != num:
            return False
        for i in range(self.level):
            if updates[i].next[i] != node:
                break
            updates[i].next[i] = node.next[i]
        while self.level > 1 and not self.head.next[self.level - 1]:
            self.level -= 1
        return True


def _test():
    random.seed(0)
    sl = Skiplist()
    sl.add(1)
    sl.add(2)
    sl.add(3)
    assert sl.search(0) is False
    sl.add(4)
    assert sl.search(1) is True
    assert sl.erase(0) is False
    assert sl.erase(1) is True
    assert sl.search(1) is False
    print("LC 1206 OK")


if __name__ == "__main__":
    _test()
