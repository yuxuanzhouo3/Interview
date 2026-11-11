# LeetCodeHard / DataStructure

每个 `*.py` 统一结构：

1. **Code template**: `query` / `update` / `add` / `del`
2. **Test case**: 覆盖上述四类操作
3. **Application**: 真实系统 / Java 对照 / 版本演进（如 B+ → LSM）

## AdvancedTree/

| # | File | Structure | 典型应用 |
|---|------|-----------|----------|
| 1 | `bst.py` | BST | 教学基线；最坏 O(n) |
| 2 | `avl.py` | AVL | 读多写少内存表 |
| 3 | `red_black.py` | Red-Black | Java `TreeMap`/`TreeSet`；C++ `std::map` |
| 4 | `b_tree.py` | B-Tree | 文件系统 / 部分 DB |
| 5 | `b_plus.py` | B+ Tree | MySQL InnoDB；范围扫 |
| 6 | `lsm.py` | LSM Tree | LevelDB / RocksDB / Cassandra；写密集 |
| 7 | `kd_tree.py` | KD-Tree | 低维 KNN / GIS |
| 8 | `bit.py` | BIT / Fenwick | 动态前缀和、逆序对 |
| 9 | `segment_tree.py` | Segment Tree | 区间改查、扫描线 |
| 10 | `trie.py` | Trie | 前缀搜索、补全 |
| 11 | `treap.py` | Treap | 竞赛随机平衡 / 隐式序列 |
| 12 | `splay.py` | Splay | 局部性热点访问 |

### 面试金句：B+ vs LSM（Redis / DB）

- **经典 Redis（内存）**: `Hash` + `SkipList(ZSET)`，不是 B+。
- **磁盘索引演进**: 旧式 RDBMS / InnoDB → **B+**（读/范围友好，就地更新）；写密集现代 KV → **LSM**（WAL+MemTable 顺序写）。
- **为何写 LSM 更好**: 随机写 → 顺序写；代价是多层读 + compaction。
- **Java 对照**: `Hashtable`/`HashMap` = 哈希无序 O(1)；`TreeMap` = 红黑有序 O(log n)。磁盘场景另论 B+/LSM。

## AdvancedGraph/

- `adjacency.py` — 邻接表 CRUD + BFS
- `union_find.py` — 并查集
- `shortest_path.py` — Dijkstra

## AdvancedStack/

- `monotonic_stack.py` — 单调栈 / Next Greater
- `min_stack.py` — O(1) getMin

## AdvancedQueue/

- `monotonic_queue.py` — 滑动窗口最值
- `circular_queue.py` — 环形队列

## AdvancedHeap/

- `binary_heap.py` — 二叉堆 / 懒删除更新
- `fibonacci_heap.py` — 斐波那契堆接口教学版

## 跑测试

```bash
cd CodePractice/LeetCodeHard/DataStructure
python AdvancedTree/bst.py
# … 或批量：
find . -name '*.py' -print0 | xargs -0 -n1 python
```
