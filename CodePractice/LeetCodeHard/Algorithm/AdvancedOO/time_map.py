"""
6. Time Based Key-Value Store
Template: query(timestamp) / update(=set) / add(set) / del(key)
App: LC 981。
"""

from collections import defaultdict
import bisect


class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)  # key -> [(ts, val)]

    def add(self, key, value, timestamp):
        self.set(key, value, timestamp)

    def set(self, key, value, timestamp):
        self.store[key].append((timestamp, value))

    def update(self, key, value, timestamp):
        self.set(key, value, timestamp)
        return True

    def query(self, key, timestamp):
        return self.get(key, timestamp)

    def get(self, key, timestamp):
        arr = self.store.get(key, [])
        # 在仅含 ts 的序列上二分右界
        ts_list = [t for t, _ in arr]
        i = bisect.bisect_right(ts_list, timestamp) - 1
        return arr[i][1] if i >= 0 else ""

    def delete(self, key):
        return self.store.pop(key, None) is not None


def _test():
    t = TimeMap()
    t.set("foo", "bar", 1)
    assert t.get("foo", 1) == "bar"
    assert t.get("foo", 3) == "bar"
    t.set("foo", "bar2", 4)
    assert t.get("foo", 4) == "bar2"
    assert t.get("foo", 5) == "bar2"
    print("TimeMap OK")


APPLICATION = """
- 同一 key 时间戳递增追加；二分找 ≤ timestamp 的右界。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
