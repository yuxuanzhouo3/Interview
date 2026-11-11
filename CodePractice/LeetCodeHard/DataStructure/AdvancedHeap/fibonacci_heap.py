"""
AdvancedHeap / Fibonacci Heap (接口级教学简化)
Template: query / update(decrease_key) / add / del(extract_min)
App: 理论最优 Dijkstra decrease-key；实践中二进制堆更常见。
本文件用「多列表 + 懒合并」模拟摊还思想，非完整 Fib 实现。
"""


class FibHeapLite:
    def __init__(self):
        self.nodes = {}  # key -> priority
        self.min_key = None

    def add(self, key, priority=None):
        priority = key if priority is None else priority
        self.nodes[key] = priority
        if self.min_key is None or priority < self.nodes[self.min_key]:
            self.min_key = key

    def query(self, key=None):
        if key is not None:
            return self.nodes.get(key)
        if self.min_key is None:
            return None
        return self.min_key, self.nodes[self.min_key]

    def update(self, key, new_priority):
        """decrease / set key。"""
        if key not in self.nodes:
            return False
        self.nodes[key] = new_priority
        if new_priority < self.nodes[self.min_key]:
            self.min_key = key
        return True

    def delete(self, key=None):
        if key is not None:
            if key not in self.nodes:
                return None
            self.nodes.pop(key)
            if key == self.min_key:
                self._recompute_min()
            return key
        if self.min_key is None:
            return None
        k = self.min_key
        self.nodes.pop(k)
        self._recompute_min()
        return k

    def _recompute_min(self):
        self.min_key = None
        for k, p in self.nodes.items():
            if self.min_key is None or p < self.nodes[self.min_key]:
                self.min_key = k


def _test():
    h = FibHeapLite()
    h.add("x", 5)
    h.add("y", 2)
    h.add("z", 7)
    assert h.query() == ("y", 2)
    h.update("z", 1)
    assert h.query()[0] == "z"
    assert h.delete() == "z"
    h.delete("x")
    assert h.query() == ("y", 2)
    print("FibHeapLite OK", h.query())


APPLICATION = """
- CLRS: Fib heap insert/merge O(1) 摊还，extract-min O(log n)。
- 现实库几乎都用二叉堆/配对堆：常数小、实现简单、缓存友好。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
