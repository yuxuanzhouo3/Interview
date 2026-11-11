"""
6. LSM Tree (Log-Structured Merge-Tree) — 教学简化版
Template: query / update / add / del
App: LevelDB, RocksDB, Cassandra, HBase, 部分 Redis 磁盘引擎。
核心: MemTable + 不可变 SSTable 层；写走顺序日志，读可能多层查找。
"""

try:
    from sortedcontainers import SortedDict as _Sorted
except ImportError:
    _Sorted = None


class MemTable:
    """内存有序表；无 sortedcontainers 时用 dict + 排序扫描。"""

    def __init__(self):
        self._d = _Sorted() if _Sorted else {}

    def put(self, k, v):
        self._d[k] = v

    def get(self, k):
        return self._d.get(k, None) if not isinstance(self._d, dict) else self._d.get(k)

    def items(self):
        if _Sorted and isinstance(self._d, _Sorted):
            return list(self._d.items())
        return sorted(self._d.items())

    def __len__(self):
        return len(self._d)

    def clear(self):
        self._d = _Sorted() if _Sorted else {}


class SSTable:
    """不可变有序文件抽象（内存 list 模拟）。tombstone = DELETE_MARK"""

    DELETE_MARK = object()

    def __init__(self, pairs):
        # pairs: sorted list[(k,v)]
        self.pairs = list(pairs)

    def get(self, key):
        lo, hi = 0, len(self.pairs) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            k, v = self.pairs[mid]
            if k == key:
                return v
            if k < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return None


class LSMTree:
    def __init__(self, memtable_limit=4):
        self.mem = MemTable()
        self.limit = memtable_limit
        self.levels = []  # list[SSTable], index0 = newest
        self._wal = []    # 简化 WAL

    # ---------- add / update（LSM 中 update=再次 put）----------
    def add(self, key, val=None):
        val = key if val is None else val
        self._wal.append(("PUT", key, val))
        self.mem.put(key, val)
        if len(self.mem) >= self.limit:
            self._flush()

    def update(self, key, val):
        if self.query(key) is None:
            return False
        self.add(key, val)
        return True

    # ---------- del（墓碑）----------
    def delete(self, key):
        self._wal.append(("DEL", key, None))
        self.mem.put(key, SSTable.DELETE_MARK)
        if len(self.mem) >= self.limit:
            self._flush()

    # ---------- query（新 → 旧）----------
    def query(self, key):
        v = self.mem.get(key)
        if v is not None:
            return None if v is SSTable.DELETE_MARK else v
        for sst in self.levels:
            v = sst.get(key)
            if v is not None:
                return None if v is SSTable.DELETE_MARK else v
        return None

    def _flush(self):
        sst = SSTable(self.mem.items())
        self.levels.insert(0, sst)
        self.mem.clear()
        if len(self.levels) >= 3:
            self._compact()

    def _compact(self):
        """简化: 合并所有 SSTable，新值覆盖旧值。"""
        merged = {}
        for sst in reversed(self.levels):
            for k, v in sst.pairs:
                merged[k] = v
        pairs = sorted((k, v) for k, v in merged.items() if v is not SSTable.DELETE_MARK)
        # 保留墓碑在更复杂实现中用于跨层；教学 compact 直接丢掉已删 key
        self.levels = [SSTable(pairs)] if pairs else []


def _test():
    t = LSMTree(memtable_limit=3)
    t.add("a", 1)
    t.add("b", 2)
    t.add("c", 3)  # flush
    assert t.query("a") == 1
    assert t.update("a", 10)
    t.add("d", 4)
    t.add("e", 5)  # may flush/compact
    assert t.query("a") == 10
    t.delete("b")
    assert t.query("b") is None
    assert t.query("c") == 3
    print("LSM OK", {k: t.query(k) for k in "abcde"})


APPLICATION = """
【真实应用】
- LevelDB / RocksDB / Cassandra / HBase / ClickHouse MergeTree 思想同源。
- Redis: 核心是内存哈希+跳表；Redis on Flash / 企业磁盘方案常嵌 RocksDB → LSM。

【为何写优于 B+】
1. 写路径: WAL + MemTable 顺序追加，避免 B+ 页级随机写。
2. Flush/Compaction 大批量顺序写盘，磁盘带宽吃满。
3. 适合 append-heavy、ingest 高的业务（日志、时序、消息、Feed）。

【代价】
- 读可能查多层 + Bloom；空间短暂放大；compaction CPU/IO。
- 点查延迟抖动；范围读不如调优好的 B+ 稳（视负载而定）。

【版本锚点】
- Google Bigtable 论文(2006) → LevelDB(2011) → RocksDB。
- MySQL InnoDB 仍 B+；写极重场景用 RocksDB/TiKV(RocksDB)/Cassandra。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
