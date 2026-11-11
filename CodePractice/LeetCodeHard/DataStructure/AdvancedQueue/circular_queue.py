"""
AdvancedQueue / Circular Queue + Deque kit
Template: query / update / add / del
App: 缓冲区、固定长度重放队列、LC 622 Design Circular Queue。
"""


class CircularQueue:
    def __init__(self, k):
        self.buf = [None] * k
        self.k = k
        self.head = 0
        self.size = 0

    def add(self, val):
        if self.size == self.k:
            return False
        tail = (self.head + self.size) % self.k
        self.buf[tail] = val
        self.size += 1
        return True

    def delete(self):
        if self.size == 0:
            return None
        v = self.buf[self.head]
        self.buf[self.head] = None
        self.head = (self.head + 1) % self.k
        self.size -= 1
        return v

    def query(self, which="front"):
        if self.size == 0:
            return None
        if which == "front":
            return self.buf[self.head]
        return self.buf[(self.head + self.size - 1) % self.k]

    def update(self, val, which="rear"):
        if self.size == 0:
            return False
        if which == "front":
            self.buf[self.head] = val
        else:
            self.buf[(self.head + self.size - 1) % self.k] = val
        return True


def _test():
    q = CircularQueue(3)
    assert q.add(1) and q.add(2) and q.add(3) and not q.add(4)
    assert q.query("front") == 1 and q.query("rear") == 3
    assert q.delete() == 1
    assert q.add(4)
    assert q.update(9, "rear") and q.query("rear") == 9
    print("CircularQueue OK", q.query("front"), q.query("rear"))


APPLICATION = """
- 内核环形缓冲、音频/日志 ring buffer、固定重连队列。
- Java ArrayDeque；Python collections.deque。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
