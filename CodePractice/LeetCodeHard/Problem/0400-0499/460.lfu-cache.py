"""
LC 460. LFU Cache — Hard
Best: freq map + doubly linked lists + key map — O(1)
"""

from collections import defaultdict, OrderedDict


class LFUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.minf = 0
        self.key_val_freq = {}
        self.freq_keys = defaultdict(OrderedDict)

    def _touch(self, key):
        val, f = self.key_val_freq[key]
        del self.freq_keys[f][key]
        if not self.freq_keys[f] and self.minf == f:
            self.minf += 1
        self.freq_keys[f + 1][key] = None
        self.key_val_freq[key] = (val, f + 1)

    def get(self, key: int) -> int:
        if key not in self.key_val_freq:
            return -1
        self._touch(key)
        return self.key_val_freq[key][0]

    def put(self, key: int, value: int) -> None:
        if self.cap == 0:
            return
        if key in self.key_val_freq:
            val, f = self.key_val_freq[key]
            self.key_val_freq[key] = (value, f)
            self._touch(key)
            return
        if len(self.key_val_freq) >= self.cap:
            old, _ = self.freq_keys[self.minf].popitem(last=False)
            del self.key_val_freq[old]
        self.key_val_freq[key] = (value, 1)
        self.freq_keys[1][key] = None
        self.minf = 1


def _test():
    c = LFUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)  # evict 2
    assert c.get(2) == -1
    assert c.get(3) == 3
    print("LC 460 OK")


if __name__ == "__main__":
    _test()
