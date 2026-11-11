"""
AdvancedQueue / Monotonic Deque (滑动窗口最值)
Template: query(max/min) / update(slide) / add(push) / del(pop_left expired)
App: LC 239 Sliding Window Maximum；实时窗口指标。
"""

from collections import deque


class MonotonicQueue:
    def __init__(self, find_max=True):
        self.find_max = find_max
        self.dq = deque()  # store values mono

    def add(self, x):
        if self.find_max:
            while self.dq and self.dq[-1] < x:
                self.dq.pop()
        else:
            while self.dq and self.dq[-1] > x:
                self.dq.pop()
        self.dq.append(x)

    def delete(self, x):
        """窗口滑出值 x 时调用。"""
        if self.dq and self.dq[0] == x:
            self.dq.popleft()

    def query(self):
        return self.dq[0] if self.dq else None

    def update(self, incoming, outgoing=None):
        if outgoing is not None:
            self.delete(outgoing)
        self.add(incoming)


def max_sliding_window(nums, k):
    dq = deque()  # indices, values decreasing
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


def _test():
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    q = MonotonicQueue()
    q.add(1)
    q.add(3)
    q.add(-1)
    assert q.query() == 3
    q.update(incoming=5, outgoing=1)
    assert q.query() == 5
    print("MonotonicQueue OK", max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))


APPLICATION = """
- 监控滑动窗口 P99/最大值；游戏技能 CD 窗口。
- vs 堆: 单调队列 O(n) 窗口最值；堆删任意元素更麻烦。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
