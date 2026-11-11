"""
5. B+ Tree
Template: query / update / add / del
App: MySQL InnoDB 聚簇索引；旧式磁盘 KV / 部分早期持久化引擎偏 B+。
叶子存全部数据且横向链表 → 范围查询极强；写是就地更新（随机 I/O）。
"""


class BPLeaf:
    def __init__(self):
        self.keys = []
        self.vals = []
        self.next = None  # leaf linked list


class BPInternal:
    def __init__(self):
        self.keys = []       # separator keys
        self.children = []   # len = len(keys)+1


class BPlusTree:
    def __init__(self, order=4):
        """order = max keys in a node (simplified)."""
        self.order = order
        self.root = BPLeaf()

    def _is_leaf(self, node):
        return isinstance(node, BPLeaf)

    # ---------- query ----------
    def query(self, key):
        node = self.root
        while not self._is_leaf(node):
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        for i, k in enumerate(node.keys):
            if k == key:
                return node.vals[i]
        return None

    def range_query(self, lo, hi):
        node = self.root
        while not self._is_leaf(node):
            i = 0
            while i < len(node.keys) and lo >= node.keys[i]:
                i += 1
            node = node.children[i]
        out = []
        while node:
            for k, v in zip(node.keys, node.vals):
                if k < lo:
                    continue
                if k > hi:
                    return out
                out.append((k, v))
            node = node.next
        return out

    # ---------- update ----------
    def update(self, key, val):
        node = self.root
        while not self._is_leaf(node):
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        for i, k in enumerate(node.keys):
            if k == key:
                node.vals[i] = val
                return True
        return False

    # ---------- add ----------
    def add(self, key, val=None):
        val = key if val is None else val
        if self.query(key) is not None:
            self.update(key, val)
            return
        split = self._insert(self.root, key, val)
        if split:
            sep, right = split
            new_root = BPInternal()
            new_root.keys = [sep]
            new_root.children = [self.root, right]
            self.root = new_root

    def _insert(self, node, key, val):
        if self._is_leaf(node):
            i = 0
            while i < len(node.keys) and node.keys[i] < key:
                i += 1
            node.keys.insert(i, key)
            node.vals.insert(i, val)
            if len(node.keys) > self.order:
                return self._split_leaf(node)
            return None
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        split = self._insert(node.children[i], key, val)
        if not split:
            return None
        sep, right = split
        node.keys.insert(i, sep)
        node.children.insert(i + 1, right)
        if len(node.keys) > self.order:
            return self._split_internal(node)
        return None

    def _split_leaf(self, leaf):
        mid = len(leaf.keys) // 2
        right = BPLeaf()
        right.keys = leaf.keys[mid:]
        right.vals = leaf.vals[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.vals = leaf.vals[:mid]
        right.next = leaf.next
        leaf.next = right
        return right.keys[0], right

    def _split_internal(self, node):
        mid = len(node.keys) // 2
        sep = node.keys[mid]
        right = BPInternal()
        right.keys = node.keys[mid + 1 :]
        right.children = node.children[mid + 1 :]
        node.keys = node.keys[:mid]
        node.children = node.children[: mid + 1]
        return sep, right

    # ---------- del ----------
    def delete(self, key):
        self._delete(self.root, key, None, 0)
        if not self._is_leaf(self.root) and not self.root.keys:
            self.root = self.root.children[0]

    def _delete(self, node, key, parent, idx_in_parent):
        if self._is_leaf(node):
            if key in node.keys:
                i = node.keys.index(key)
                node.keys.pop(i)
                node.vals.pop(i)
            # underflow handling simplified: accept underfull for teaching
            return
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        self._delete(node.children[i], key, node, i)
        # keep separators roughly in sync
        child = node.children[i]
        if self._is_leaf(child) and child.keys:
            if i > 0:
                node.keys[i - 1] = child.keys[0]

    def keys(self):
        return [k for k, _ in self.range_query(float("-inf"), float("inf"))]


def _test():
    t = BPlusTree(order=3)
    for k in [5, 8, 1, 7, 3, 12, 9, 6]:
        t.add(k, k * 10)
    assert t.query(7) == 70
    assert t.update(7, 77) and t.query(7) == 77
    assert [k for k, _ in t.range_query(3, 9)] == [3, 5, 6, 7, 8, 9]
    t.delete(7)
    assert t.query(7) is None
    print("B+ OK", t.keys())


APPLICATION = """
【真实应用】
- MySQL InnoDB: 聚簇 B+ 索引，主键叶子存整行；二级索引叶子存主键。
- 旧式磁盘关系库 / 很多「读多、范围扫」系统首选 B+。

【与 Redis / LSM 的关系（面试常问）】
- 经典 Redis（内存）: 哈希表 + 跳表(ZSET)，不是 B+。
- 持久化/磁盘型 KV 演进叙事:
  * 传统 RDBMS / 早期磁盘索引 → B+（读友好、范围扫友好，就地更新）。
  * 写密集现代引擎（LevelDB / RocksDB / Cassandra / 部分 Redis-on-Flash）→ LSM。
- 为何写场景 LSM 优于 B+？
  * B+: 更新落在树页上 → 大量随机写 + 写放大来自页分裂。
  * LSM: 先写 WAL/MemTable（顺序），刷盘 SSTable 也顺序写；随机写变顺序写，吞吐更高。
  * 代价: 读需跨多层/布隆过滤；需要 compaction。

版本记忆锚点（面试口述）:
  - InnoDB B+ : MySQL 5.x/8.x 一脉相承。
  - LevelDB(2011) / RocksDB(Facebook) : LSM 普及。
  - Redis 主路径仍内存结构；磁盘扩展常借 RocksDB(LSM)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
