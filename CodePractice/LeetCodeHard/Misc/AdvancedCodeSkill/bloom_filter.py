"""
CodeSkill / Bloom Filter
Template: query(maybe_contains) / update(rebuild) / add / del(unsupported)
App: 缓存穿透防护；「可能在 / 一定不在」；Redis Bloom 模块同思想。
"""

import hashlib


class BloomFilter:
    def __init__(self, size=1024, hash_count=3):
        self.size = size
        self.k = hash_count
        self.bits = 0  # 用 int 位图教学；生产用 bytearray

    def _hashes(self, item):
        s = str(item).encode()
        for i in range(self.k):
            h = hashlib.md5(s + str(i).encode()).hexdigest()
            yield int(h, 16) % self.size

    def add(self, item):
        for h in self._hashes(item):
            self.bits |= 1 << h

    def update(self, size=None, hash_count=None):
        """重建空过滤器（丢内容）。"""
        if size is not None:
            self.size = size
        if hash_count is not None:
            self.k = hash_count
        self.bits = 0
        return True

    def delete(self, item=None):
        """标准 Bloom 不支持删；返回 False。"""
        return False

    def query(self, item):
        """True = 可能存在；False = 一定不存在。"""
        return all(self.bits & (1 << h) for h in self._hashes(item))


def _test():
    b = BloomFilter(256, 3)
    b.add("apple")
    b.add("banana")
    assert b.query("apple") is True
    assert b.query("cherry") is False  # 高概率
    assert b.delete("apple") is False
    assert b.update(512, 4)
    assert b.query("apple") is False  # 已重建
    b.add("apple")
    assert b.query("apple")
    print("Bloom OK", b.query("apple"), b.query("zzz"))


APPLICATION = """
- 误判率随 bit 数与哈希数下降；不能回答「一定在」。
- 需删除 → Counting Bloom / Cuckoo Filter。
- 与 LRU 搭配：Bloom 挡穿透，LRU 挡热点。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
