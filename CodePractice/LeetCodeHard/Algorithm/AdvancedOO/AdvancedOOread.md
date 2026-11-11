# AdvancedOO · Syllabus（OOP 设计题）

**OO = Object-Oriented Design**。对照 `hard.py`「数据结构设计」：哈希 + 链表 / 堆 / 有序结构组合。面试常考：**接口清晰、均摊 O(1)、边界与淘汰策略**。

> 原贪心 & 区间调度案例已迁至 `../AdvancedGreedy/`。

## 学习路径

```
LRU（哈希+双向链表） → LFU（哈希+频次桶）
  → AllOne / FreqStack → InsertDeleteGetRandom
  → TimeMap / MedianFinder → Twitter / HitCounter
  → BloomFilter（概率结构）
```

## 文件大纲

| # | 文件 | 主题 | 必会点 | 复杂度直觉 | 典型题 |
|---|------|------|--------|------------|--------|
| 1 | `lru_cache.py` | LRU 缓存 | `dict` + 双向链表；get/put 搬到头 | O(1) | 146 |
| 2 | `lfu_cache.py` | LFU 缓存 | key→freq；freq→LRU 链表；min_freq | O(1) | 460 |
| 3 | `all_one.py` | All O'one | 计数桶双向链表；max/min key | O(1) | 432 |
| 4 | `freq_stack.py` | 最大频次栈 | freq→栈；分组弹最高频 | O(1) | 895 |
| 5 | `randomized_set.py` | O(1) 插删随机 | 数组 + 下标哈希；尾交换删 | O(1) | 380 |
| 6 | `time_map.py` | 时间键值 | key→[(ts,val)] 二分 | O(log n) | 981 |
| 7 | `median_finder.py` | 数据流中位数 | 大小根堆对顶 | O(log n) | 295 |
| 8 | `twitter.py` | 设计推特 | 关注图 + 推文链表/堆合并 | post O(1) get O(k log u) | 355 |
| 9 | `hit_counter.py` | 击中计数 | 秒级环数组 / 队列滑窗 | O(1)~O(W) | 362 |
| 10 | `BloomFilter.py` | 布隆过滤器 | 多哈希位图；Counting 可删 | O(k) | 缓存穿透 |

## 组合套路怎么选

| 需求 | 组合 |
|------|------|
| 按「最近使用」淘汰 | HashMap + 双向链表（LRU） |
| 按「频率」淘汰，同频再 LRU | HashMap + freq 桶链表（LFU） |
| 动态计数的极值 key | 计数 ↔ 双向桶（AllOne） |
| O(1) 随机取元素 | 数组 + HashMap（尾交换） |
| 历史时间点查询 | HashMap + 有序时间戳 |
| 动态中位数 | 对顶堆 |

## 面试口述 checklist

1. 先定 **接口与复杂度承诺**，再选结构，最后画指针/桶图。
2. LRU：哑头尾节点省空判；`put` 已存在要先摘再插。
3. LFU：维护 `min_freq`；同频用链表保证 LRU；freq 升桶时旧桶空则删。
4. 随机集合删除：与末尾交换再 `pop`，同步改哈希下标。
5. 说明线程安全：本仓库为单线程教学版；生产加锁或用 `ConcurrentHashMap` 思路。

## 自测

```bash
cd AdvancedOO && for f in *.py; do python3 "$f"; done
```
