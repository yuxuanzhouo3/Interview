# LeetCodeHard / DataStructure

每个 `*.py` 统一结构：

1. **Code template**: `query` / `update` / `add` / `del`
2. **Test case**: 覆盖上述四类操作
3. **Application**: 真实系统 / Java 对照 / 版本演进（如 B+ → LSM）

各子目录有完整 syllabus（学习路径 + 文件大纲 + 选型 + 面试 checklist）：

| 目录 | Syllabus |
|------|----------|
| AdvancedTree | [AdvancedTreeread.md](AdvancedTree/AdvancedTreeread.md) |
| AdvancedGraph | [AdvancedGraphread.md](AdvancedGraph/AdvancedGraphread.md) |
| AdvancedHeap | [AdvancedHeapread.md](AdvancedHeap/AdvancedHeapread.md) |
| AdvancedQueue | [AdvancedQueueread.md](AdvancedQueue/AdvancedQueueread.md) |
| AdvancedStack | [AdvancedStackread.md](AdvancedStack/AdvancedStackread.md) |
| AdvancedString | [AdvancedStringread.md](AdvancedString/AdvancedStringread.md) |

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

| # | File | Structure | 典型应用 |
|---|------|-----------|----------|
| 1 | `bipartite.py` | 二分图判定 + 最大匹配 | LC 785；任务分配；König |
| 2 | `network_flow.py` | Dinic 最大流 | 最小割、吞吐、匹配建流 |
| 3 | `adjacency.py` | 邻接表 + BFS | 通用图建模 |
| 4 | `union_find.py` | 并查集 DSU | 连通分量、Kruskal |
| 5 | `shortest_path.py` | Dijkstra | 路由 / LC 743 |
| 6 | `topo_sort.py` | Kahn 拓扑 | 课程表 / LC 210 |
| 7 | `scc.py` | Tarjan SCC | 缩点、2-SAT |
| 8 | `bridge_cut.py` | 桥 / 割点 | LC 1192 |
| 9 | `zero_one_bfs.py` | 0-1 BFS | 边权仅 0/1 |
| 10 | `bellman_ford.py` | Bellman-Ford | 负权 / 负环 |
| 11 | `floyd_warshall.py` | Floyd 全源 | 稠密小图 / 传递闭包 |
| 12 | `astar.py` | A* 网格 | 单源单汇启发搜索 |
| 13 | `mst.py` | Kruskal + Prim | LC 1135 / 1584 |
| 14 | `cycle.py` | 有向/无向环检测 | LC 207 / 684 |
| 15 | `dag_dp.py` | DAG 最长/最短路 | 关键路径 |
| 16 | `grid_graph.py` | 网格 BFS + 连通块 | LC 200；超市网格 |

模板来源对照：`CodePractice/graph.py`。

## AdvancedStack/

- `monotonic_stack.py` — 单调栈 / Next Greater
- `min_stack.py` — O(1) getMin

## AdvancedQueue/

- `monotonic_queue.py` — 滑动窗口最值
- `circular_queue.py` — 环形队列

## AdvancedHeap/

- `binary_heap.py` — 二叉堆 / 懒删除更新
- `fibonacci_heap.py` — 斐波那契堆接口教学版
- `leftist_heap.py` — 左偏树（可并堆）

## AdvancedString/

| # | File | Algorithm | 典型应用 |
|---|------|-----------|----------|
| 1 | `kmp.py` | KMP | 单模式匹配；周期串 |
| 2 | `z_algorithm.py` | Z-Algorithm | 扩展 KMP；自身匹配 |
| 3 | `rabin_karp.py` | Rabin-Karp | 滚动哈希；多模式初筛 |
| 4 | `manacher.py` | Manacher | 最长回文子串 O(n) |
| 5 | `aho_corasick.py` | Aho-Corasick | 敏感词 / 多模式 |
| 6 | `suffix_array.py` | Suffix Array + LCP | 最长重复子串；子串查询 |
| 7 | `suffix_automaton.py` | SAM | 出现次数；本质不同子串 |
| 8 | `string_hash_lcp.py` | 双哈希 + LCP | 快速判等；二分 LCP |

对照：单模式 → KMP/Z/RK；多模式 → AC；回文 → Manacher；索引级 → SA/SAM。

## 跑测试

```bash
cd CodePractice/LeetCodeHard/DataStructure
python AdvancedTree/bst.py
# … 或批量：
find . -name '*.py' -print0 | xargs -0 -n1 python
```
