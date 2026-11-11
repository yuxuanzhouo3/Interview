"""
3. All O'one Data Structure
Template: query(max/min) / update(inc/dec) / add(inc new) / del(dec→0)
App: LC 432。
"""

from collections import defaultdict


class Node:
    __slots__ = ("count", "keys", "prev", "next")

    def __init__(self, count=0):
        self.count = count
        self.keys = set()
        self.prev = self.next = None


class AllOne:
    def __init__(self):
        self.head, self.tail = Node(0), Node(0)
        self.head.next, self.tail.prev = self.tail, self.head
        self.map = {}  # key -> Node

    def _insert_after(self, node, new):
        nxt = node.next
        node.next = new
        new.prev, new.next = node, nxt
        nxt.prev = new

    def _remove(self, node):
        if node.keys:
            return
        node.prev.next = node.next
        node.next.prev = node.prev

    def add(self, key):
        self.inc(key)

    def update(self, key, delta=1):
        if delta > 0:
            self.inc(key)
        else:
            self.dec(key)
        return True

    def inc(self, key):
        if key not in self.map:
            if self.head.next.count != 1:
                self._insert_after(self.head, Node(1))
            self.head.next.keys.add(key)
            self.map[key] = self.head.next
            return
        cur = self.map[key]
        cur.keys.remove(key)
        nxt = cur.next
        if nxt is self.tail or nxt.count != cur.count + 1:
            nxt = Node(cur.count + 1)
            self._insert_after(cur, nxt)
        nxt.keys.add(key)
        self.map[key] = nxt
        self._remove(cur)

    def dec(self, key):
        if key not in self.map:
            return
        cur = self.map[key]
        cur.keys.remove(key)
        if cur.count == 1:
            del self.map[key]
        else:
            pre = cur.prev
            if pre is self.head or pre.count != cur.count - 1:
                pre = Node(cur.count - 1)
                self._insert_after(cur.prev, pre)
            pre.keys.add(key)
            self.map[key] = pre
        self._remove(cur)

    def query(self, which="max"):
        if which == "max":
            return self.getMaxKey()
        return self.getMinKey()

    def getMaxKey(self):
        if self.tail.prev is self.head:
            return ""
        return next(iter(self.tail.prev.keys))

    def getMinKey(self):
        if self.head.next is self.tail:
            return ""
        return next(iter(self.head.next.keys))

    def delete(self, key):
        while key in self.map:
            self.dec(key)


def _test():
    a = AllOne()
    a.inc("hello")
    a.inc("hello")
    assert a.getMaxKey() == "hello" and a.getMinKey() == "hello"
    a.inc("leet")
    assert a.getMaxKey() == "hello" and a.getMinKey() == "leet"
    a.dec("hello")
    a.dec("hello")
    assert a.getMaxKey() == "leet"
    print("AllOne OK")


APPLICATION = """
- 桶按 count 双向链；inc/dec 只在相邻桶挪 key。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
