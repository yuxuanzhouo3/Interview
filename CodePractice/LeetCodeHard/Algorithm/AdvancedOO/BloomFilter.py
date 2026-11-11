"""
BloomFilter — 概率型成员查询（OO 目录额外结构题）
Template: query / update(=add) / add / del(不支持真删，可用 counting)
App: Redis Bloom、爬虫 URL 去重、缓存穿透防护。
"""

try:
    import mmh3 as _mmh3
except ImportError:
    _mmh3 = None


def _hashes(item, k, m):
    s = str(item)
    if _mmh3:
        for i in range(k):
            yield _mmh3.hash(s, i) % m
        return
    # fallback: 双哈希派生
    h1 = hash(s) & 0xFFFFFFFF
    h2 = hash(s + "#") & 0xFFFFFFFF or 0x9E3779B9
    for i in range(k):
        yield (h1 + i * h2) % m


class BloomFilter:
    def __init__(self, m=1024, k=3):
        self.m, self.k = m, k
        self.bits = 0  # 用 int 位图（m 不宜过大；教学用）

    def add(self, item):
        for h in _hashes(item, self.k, self.m):
            self.bits |= 1 << h

    def update(self, item):
        """同 add（Bloom 无改值语义）。"""
        self.add(item)
        return True

    def query(self, item):
        """可能假阳；无假阴（未删前提下）。"""
        return all(self.bits >> h & 1 for h in _hashes(item, self.k, self.m))

    def delete(self, item):
        """标准 Bloom 不支持删除；返回 False。见 CountingBloom。"""
        return False


class CountingBloomFilter:
    def __init__(self, m=256, k=3):
        self.m, self.k = m, k
        self.cnt = [0] * m

    def add(self, item):
        for h in _hashes(item, self.k, self.m):
            self.cnt[h] += 1

    def delete(self, item):
        if not self.query(item):
            return False
        for h in _hashes(item, self.k, self.m):
            if self.cnt[h] == 0:
                return False
        for h in _hashes(item, self.k, self.m):
            self.cnt[h] -= 1
        return True

    def query(self, item):
        return all(self.cnt[h] > 0 for h in _hashes(item, self.k, self.m))

    def update(self, old, new):
        self.delete(old)
        self.add(new)
        return True


def _test():
    bf = BloomFilter(m=512, k=4)
    for x in ["apple", "banana", "cherry"]:
        bf.add(x)
    assert bf.query("apple") and bf.query("banana")
    assert bf.delete("apple") is False
    cbf = CountingBloomFilter()
    cbf.add("a")
    cbf.add("b")
    assert cbf.query("a") and cbf.delete("a") and not cbf.query("a")
    assert cbf.query("b")
    print("BloomFilter OK")


APPLICATION = """
- RedisBloom / Guava BloomFilter；缓存穿透：先查 Bloom 再查 DB。
- 有假阳无假阴；需要删除 → Counting Bloom / Cuckoo Filter。
- Java Guava: BloomFilter.create(Funnels..., expectedInsertions, fpp)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
