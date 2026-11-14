"""
LC 432. All O`one Data Structure — Hard
Best: Doubly linked freq buckets + key map — O(1)
"""

from collections import defaultdict


class Node:
    __slots__ = ("freq", "keys", "prev", "next")
    def __init__(self, freq=0):
        self.freq = freq
        self.keys = set()
        self.prev = self.next = None


class AllOne:
    def __init__(self):
        self.head = Node(0)
        self.tail = Node(0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.map = {}  # key -> node

    def _add_after(self, node, freq):
        nxt = Node(freq)
        nxt.prev, nxt.next = node, node.next
        node.next.prev = nxt
        node.next = nxt
        return nxt

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def inc(self, key: str) -> None:
        if key not in self.map:
            # freq 1 after head
            if self.head.next.freq != 1:
                self._add_after(self.head, 1)
            node = self.head.next
            node.keys.add(key)
            self.map[key] = node
            return
        cur = self.map[key]
        cur.keys.remove(key)
        freq = cur.freq + 1
        if cur.next.freq != freq:
            self._add_after(cur, freq)
        nxt = cur.next
        nxt.keys.add(key)
        self.map[key] = nxt
        if not cur.keys:
            self._remove(cur)

    def dec(self, key: str) -> None:
        cur = self.map[key]
        cur.keys.remove(key)
        if cur.freq == 1:
            del self.map[key]
        else:
            freq = cur.freq - 1
            if cur.prev.freq != freq:
                self._add_after(cur.prev, freq)
            prev = cur.prev
            prev.keys.add(key)
            self.map[key] = prev
        if not cur.keys:
            self._remove(cur)

    def getMaxKey(self) -> str:
        if self.tail.prev is self.head:
            return ""
        return next(iter(self.tail.prev.keys))

    def getMinKey(self) -> str:
        if self.head.next is self.tail:
            return ""
        return next(iter(self.head.next.keys))


def _test():
    a = AllOne()
    a.inc("hello")
    a.inc("hello")
    assert a.getMaxKey() == "hello"
    assert a.getMinKey() == "hello"
    a.inc("leet")
    assert a.getMaxKey() == "hello"
    assert a.getMinKey() == "leet"
    print("LC 432 OK")


if __name__ == "__main__":
    _test()
