# AdvancedTree · Syllabus

统一接口：`query` / `update` / `add` / `del`。按「内存平衡树 → 磁盘索引 → 写优化 → 区间结构 → 特殊树」推进。

## 学习路径

```
BST → AVL → RedBlack → Treap/Splay
  → B / B+ → LSM
  → BIT / SegmentTree
  → Trie / KD
```

## 文件大纲

| # | 文件 | 结构 | 必会点 | 复杂度 | 典型应用 |
|---|------|------|--------|--------|----------|
| 1 | `bst.py` | 二叉搜索树 | 查找/插入/删（后继替换） | 均 O(log n)，坏 O(n) | 教学基线 |
| 2 | `avl.py` | AVL | 平衡因子、四种旋转 | 严格 O(log n) | 读多写少内存表 |
| 3 | `red_black.py` | 红黑树 | 染色+旋转、近似平衡 | O(log n)，写旋转少 | Java `TreeMap`/`TreeSet` |
| 4 | `b_tree.py` | B 树 | 多路分裂/合并 | 磁盘页友好 | 文件系统、部分 DB |
| 5 | `b_plus.py` | B+ 树 | 数据全在叶子+链表 | 范围扫极强 | MySQL InnoDB |
| 6 | `lsm.py` | LSM | MemTable→SSTable、墓碑、compact | 写顺序化 | RocksDB / Cassandra |
| 7 | `kd_tree.py` | KD 树 | 轴轮换、近邻剪枝 | 低维好，高维崩 | GIS、KNN |
| 8 | `bit.py` | 树状数组 | `i+=i&-i` 前缀和 | O(log n) | 逆序对、LC 307/315 |
| 9 | `segment_tree.py` | 线段树 | 建树/懒标记区间加 | O(log n) | 区间最值、扫描线 |
| 10 | `trie.py` | 字典树 | 共享前缀 | O(L) | 补全、最长前缀匹配 |
| 11 | `treap.py` | Treap | split/merge、随机堆优先级 | 期望 O(log n) | 竞赛、隐式序列 |
| 12 | `splay.py` | Splay | 访问旋到根 | 摊还 O(log n) | 局部性热点 |

## 选型对照（面试高频）

| 需求 | 选 | 别选原因 |
|------|-----|----------|
| Java 有序 Map | `TreeMap` = 红黑 | 不是 HashMap，也不是裸 BST |
| MySQL 聚簇索引 | B+ | 范围扫 + 页对齐 |
| 写密集磁盘 KV | LSM | B+ 就地更新随机写多 |
| 动态前缀和 | BIT | 更短；复杂区间操作用线段树 |
| 区间加 + 区间和 | Segment Tree + lazy | BIT 懒标记不自然 |
| 前缀搜索 | Trie | Hash 只能精确键 |
| 低维近邻 | KD | 高维改 LSH/向量索引 |

## B+ vs LSM（金句）

- **B+**：读/范围友好；更新落页 → **随机写**。
- **LSM**：WAL + MemTable → Flush SSTable → **顺序写**；读可能跨多层 + compaction。
- **Redis 内存主路径**：Hash + SkipList；磁盘扩展常借 RocksDB（LSM），不是内核从 B+ 换成 LSM。

## 内存平衡树怎么记

| | AVL | RB | Treap | Splay |
|--|-----|----|-------|-------|
| 平衡 | 最严 | 近似 | 随机期望 | 摊还 |
| 写成本 | 旋转多 | 工程甜点 | 实现短 | 局部性好 |
| 生产 | 少 | 极多 | 竞赛多 | 少（并发难） |

## 自测

```bash
cd AdvancedTree && for f in *.py; do python3 "$f"; done
```
