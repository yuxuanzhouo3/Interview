"""
1. LRU Cache — HashMap + 双向链表
Template: query(get) / update(put overwrite) / add(put) / del(evict)
App: LC 146；Redis maxmemory-policy allkeys-lru；CPU cache 思想。
"""


class Node:
    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}
        self.head, self.tail = Node(), Node()
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        nxt = self.head.next
        self.head.next = node
        node.prev, node.next = self.head, nxt
        nxt.prev = node

    def _move_front(self, node):
        self._remove(node)
        self._add_front(node)

    def query(self, key):
        return self.get(key)

    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._move_front(node)
        return node.val

    def update(self, key, val):
        if key not in self.map:
            return False
        self.put(key, val)
        return True

    def add(self, key, val):
        self.put(key, val)

    def put(self, key, val):
        if key in self.map:
            node = self.map[key]
            node.val = val
            self._move_front(node)
            return
        node = Node(key, val)
        self.map[key] = node
        self._add_front(node)
        if len(self.map) > self.cap:
            self.delete()

    def delete(self, key=None):
        """delete() 淘汰尾部；delete(key) 显式删。"""
        if key is not None:
            if key not in self.map:
                return None
            node = self.map.pop(key)
            self._remove(node)
            return node.val
        if self.tail.prev is self.head:
            return None
        node = self.tail.prev
        self._remove(node)
        del self.map[node.key]
        return node.key


def _test():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)  # evict 2
    assert c.get(2) == -1
    c.put(4, 4)  # evict 1
    assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
    assert c.update(4, 40) and c.get(4) == 40
    print("LRU OK")


APPLICATION = """
- get/put 都将节点移到最前；超容删尾。
- Java: LinkedHashMap(accessOrder=true)；手写双链表更稳控细节。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
