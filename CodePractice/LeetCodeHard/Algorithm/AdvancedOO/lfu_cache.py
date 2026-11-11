"""
2. LFU Cache — key↔freq + freq 桶内 LRU 链表
Template: query(get) / update / add(put) / del(evict)
App: LC 460；热点缓存按访问频次淘汰。
"""

from collections import defaultdict, OrderedDict


class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.min_f = 0
        self.key_val = {}
        self.key_freq = {}
        self.freq_keys = defaultdict(OrderedDict)  # freq -> ordered keys (LRU)

    def _touch(self, key):
        f = self.key_freq[key]
        del self.freq_keys[f][key]
        if not self.freq_keys[f] and self.min_f == f:
            self.min_f += 1
        self.key_freq[key] = f + 1
        self.freq_keys[f + 1][key] = None

    def query(self, key):
        return self.get(key)

    def get(self, key):
        if key not in self.key_val or self.cap == 0:
            return -1
        self._touch(key)
        return self.key_val[key]

    def update(self, key, val):
        if key not in self.key_val:
            return False
        self.put(key, val)
        return True

    def add(self, key, val):
        self.put(key, val)

    def put(self, key, val):
        if self.cap == 0:
            return
        if key in self.key_val:
            self.key_val[key] = val
            self._touch(key)
            return
        if len(self.key_val) >= self.cap:
            self.delete()
        self.key_val[key] = val
        self.key_freq[key] = 1
        self.freq_keys[1][key] = None
        self.min_f = 1

    def delete(self, key=None):
        if key is not None:
            if key not in self.key_val:
                return None
            f = self.key_freq.pop(key)
            del self.freq_keys[f][key]
            del self.key_val[key]
            return key
        # evict LFU then LRU among that freq
        k, _ = self.freq_keys[self.min_f].popitem(last=False)
        del self.key_val[k]
        del self.key_freq[k]
        return k


def _test():
    c = LFUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1  # freq1=2, freq2=1
    c.put(3, 3)  # evict key 2
    assert c.get(2) == -1 and c.get(3) == 3
    assert c.get(1) == 1
    c.put(4, 4)  # evict 3 (freq1)
    assert c.get(3) == -1 and c.get(4) == 4 and c.get(1) == 1
    print("LFU OK")


APPLICATION = """
- 同频内再按 LRU；维护 min_freq 避免扫桶。
- vs LRU: LFU 抗偶发扫描更稳，实现更重。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
