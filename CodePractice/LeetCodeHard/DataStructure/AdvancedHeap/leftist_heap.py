"""
AdvancedHeap / Leftist Heap (可并堆)
Template: query / update(re-merge) / add / del / merge
App: 需要高效 merge 的优先队列；竞赛可并堆。
"""


class Node:
    def __init__(self, key, val=None):
        self.key = key  # priority
        self.val = val if val is not None else key
        self.left = None
        self.right = None
        self.dist = 0  # null-path length


def _dist(n):
    return -1 if not n else n.dist


def merge(a, b):
    if not a or not b:
        return a or b
    if a.key > b.key:
        a, b = b, a
    a.right = merge(a.right, b)
    if _dist(a.left) < _dist(a.right):
        a.left, a.right = a.right, a.left
    a.dist = _dist(a.right) + 1
    return a


class LeftistHeap:
    def __init__(self):
        self.root = None

    def add(self, key, val=None):
        self.root = merge(self.root, Node(key, val))

    def query(self):
        return None if not self.root else (self.root.key, self.root.val)

    def delete(self):
        if not self.root:
            return None
        v = self.root.val
        self.root = merge(self.root.left, self.root.right)
        return v

    def update(self, old_key, new_key, val=None):
        """简化: 抽出后重插（教学）。"""
        items = []
        found = False

        def collect(n):
            nonlocal found
            if not n:
                return
            if not found and n.key == old_key and (val is None or n.val == val):
                found = True
                items.append((new_key, n.val if val is None else val))
            else:
                items.append((n.key, n.val))
            collect(n.left)
            collect(n.right)

        collect(self.root)
        self.root = None
        for k, v in items:
            self.add(k, v)
        return found

    def merge_with(self, other):
        self.root = merge(self.root, other.root)
        other.root = None


def _test():
    h = LeftistHeap()
    for x in [5, 1, 3, 4]:
        h.add(x, x)
    assert h.query() == (1, 1)
    assert h.delete() == 1
    assert h.update(5, 0)
    assert h.query()[0] == 0
    h2 = LeftistHeap()
    h2.add(-1)
    h.merge_with(h2)
    assert h.query()[0] == -1
    print("LeftistHeap OK", h.query())


APPLICATION = """
- 可并优先队列：多路归并、并行调度合并事件堆。
- vs 二叉堆: 二叉堆 merge 是 O(n)；左偏树 merge O(log n)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
