"""
CodeSkill / LRU Cache（哈希 + 双向链表）
Template: query(get) / update(put) / add / del
App: LC 146；设计题基本功。
"""


class Node:
    __slots__ = ("k", "v", "prev", "next")

    def __init__(self, k=0, v=0):
        self.k, self.v = k, v
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity=2):
        self.cap = capacity
        self.map = {}
        self.head, self.tail = Node(), Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def add(self, capacity):
        """重置容量（不清理已有，仅改 cap）。"""
        self.cap = capacity

    def update(self, key, value):
        return self.put(key, value)

    def delete(self, key):
        if key not in self.map:
            return False
        node = self.map.pop(key)
        self._remove(node)
        return True

    def query(self, key):
        return self.get(key)

    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._add_front(node)
        return node.v

    def put(self, key, value):
        if key in self.map:
            node = self.map[key]
            node.v = value
            self._remove(node)
            self._add_front(node)
            return True
        if len(self.map) >= self.cap:
            lru = self.tail.prev
            self._remove(lru)
            self.map.pop(lru.k)
        node = Node(key, value)
        self.map[key] = node
        self._add_front(node)
        return True


def _test():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.query(1) == 1
    c.update(3, 3)  # evict 2
    assert c.query(2) == -1
    assert c.delete(1)
    c.add(3)
    c.put(4, 4)
    print("LRU OK", c.query(3), c.query(4))


APPLICATION = """
- HashMap: key→node；链表维护新近顺序。
- LFU: 额外频次→链表桶；面试常跟 460。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
