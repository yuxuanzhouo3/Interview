# AdvancedHeap · Syllabus

统一接口：`query` / `update` / `add` / `del`（及可并堆的 `merge`）。

## 学习路径

```
BinaryHeap（主力） → LeftistHeap（可并） → FibonacciHeap（理论）
```

## 文件大纲

| # | 文件 | 结构 | 必会点 | 复杂度要点 | 典型应用 |
|---|------|------|--------|------------|----------|
| 1 | `binary_heap.py` | 二叉堆 | sift up/down；懒删除更新 key | insert/pop O(log n) | Dijkstra、TopK、任务调度 |
| 2 | `leftist_heap.py` | 左偏树 | NPL、merge 保左偏 | merge O(log n) | 多路归并、可并优先队列 |
| 3 | `fibonacci_heap.py` | 斐波那契堆（教学接口） | decrease-key 摊还优 | 理论 Dijkstra 更优 | 教材；实践少用 |

## 选型

| 场景 | 选 |
|------|----|
| 日常优先队列 / LC | Binary Heap（`heapq` / `PriorityQueue`） |
| 需要频繁 merge 两堆 | Leftist / Skew / Pairing |
| 考试默写 decrease-key 摊还 | Fibonacci（口述即可，实现用二叉堆） |

## 面试 checklist

1. 堆是完全二叉树 → 数组下标父子关系。
2. 更新优先级：真 decrease-key（需句柄）或懒插入+过期弹出（本仓库 binary 版）。
3. Java `PriorityQueue` 默认小根堆；大根堆取反或自定义 Comparator。
4. TopK：大小为 K 的堆，别全排序。

## 自测

```bash
cd AdvancedHeap && for f in *.py; do python3 "$f"; done
```
