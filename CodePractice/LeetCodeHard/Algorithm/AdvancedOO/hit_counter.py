"""
9. Design Hit Counter（最近 5 分钟）
Template: query(getHits) / update / add(hit) / del(过期清理)
App: LC 362；监控 QPS 滑窗。
"""

from collections import deque


class HitCounter:
    def __init__(self, window=300):
        self.window = window
        self.q = deque()  # timestamps

    def add(self, timestamp):
        self.hit(timestamp)

    def hit(self, timestamp):
        self.q.append(timestamp)
        self._evict(timestamp)

    def update(self, timestamp):
        self.hit(timestamp)
        return True

    def query(self, timestamp):
        return self.getHits(timestamp)

    def getHits(self, timestamp):
        self._evict(timestamp)
        return len(self.q)

    def delete(self, timestamp=None):
        if timestamp is None:
            self.q.clear()
            return
        self._evict(timestamp)

    def _evict(self, timestamp):
        while self.q and self.q[0] <= timestamp - self.window:
            self.q.popleft()


class HitCounterRing:
    """秒级环数组 O(1) 空间固定 300。"""

    def __init__(self):
        self.times = [0] * 300
        self.hits = [0] * 300

    def hit(self, timestamp):
        i = timestamp % 300
        if self.times[i] != timestamp:
            self.times[i] = timestamp
            self.hits[i] = 1
        else:
            self.hits[i] += 1

    def getHits(self, timestamp):
        return sum(
            h for t, h in zip(self.times, self.hits) if timestamp - t < 300
        )

    add = hit
    query = getHits

    def update(self, timestamp):
        self.hit(timestamp)
        return True

    def delete(self):
        self.times = [0] * 300
        self.hits = [0] * 300


def _test():
    h = HitCounter()
    h.hit(1)
    h.hit(2)
    h.hit(3)
    assert h.getHits(4) == 3
    h.hit(300)
    assert h.getHits(300) == 4
    assert h.getHits(301) == 3
    r = HitCounterRing()
    r.hit(1)
    r.hit(2)
    assert r.getHits(3) == 2
    print("HitCounter OK")


APPLICATION = """
- 队列版直观；环数组版空间 O(W)、适合固定窗秒级。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
