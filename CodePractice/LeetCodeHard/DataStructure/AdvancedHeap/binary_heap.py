"""
AdvancedHeap / Binary Heap (min-heap)
Template: query(peek) / update(decrease/increase key) / add(push) / del(pop/remove)
App: 优先队列、Dijkstra、TopK、Java PriorityQueue、Python heapq。
"""

import heapq


class BinaryHeap:
    def __init__(self):
        self.h = []
        self._entry = {}  # key -> val (for update tracking, simplified unique keys)

    def add(self, key, priority=None):
        priority = key if priority is None else priority
        heapq.heappush(self.h, (priority, key))
        self._entry[key] = priority

    def query(self, key=None):
        if key is None:
            while self.h and self._entry.get(self.h[0][1]) != self.h[0][0]:
                heapq.heappop(self.h)  # stale
            return None if not self.h else (self.h[0][1], self.h[0][0])
        return self._entry.get(key)

    def update(self, key, new_priority):
        if key not in self._entry:
            return False
        self._entry[key] = new_priority
        heapq.heappush(self.h, (new_priority, key))  # lazy
        return True

    def delete(self, key=None):
        """delete() pop min; delete(key) lazy invalidate。"""
        if key is not None:
            if key not in self._entry:
                return None
            self._entry.pop(key)
            return key
        while self.h:
            p, k = heapq.heappop(self.h)
            if self._entry.get(k) == p:
                self._entry.pop(k, None)
                return k
        return None


def _test():
    h = BinaryHeap()
    for k, p in [("a", 3), ("b", 1), ("c", 2)]:
        h.add(k, p)
    assert h.query() == ("b", 1)
    assert h.update("a", 0)
    assert h.query()[0] == "a"
    assert h.delete() == "a"
    h.delete("c")
    assert h.query() == ("b", 1)
    print("BinaryHeap OK", h.query())


APPLICATION = """
- 任务调度、合并 K 有序、TopK 高频 LC 347。
- Java PriorityQueue 默认小根堆；TreeMap 可当有序优先但常数不同。
- 斐波那契堆理论 decrease-key 更优，实践少用 → 见 fibonacci_heap.py。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
