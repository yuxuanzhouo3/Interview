# AdvancedQueue · Syllabus

统一接口：`query` / `update` / `add` / `del`。

## 学习路径

```
CircularQueue（基础缓冲） → MonotonicQueue（滑动窗口最值）
```

## 文件大纲

| # | 文件 | 结构 | 必会点 | 复杂度 | 典型应用 |
|---|------|------|--------|--------|----------|
| 1 | `circular_queue.py` | 环形队列 | head/size 取模、满/空判定 | O(1) | LC 622、ring buffer、日志缓冲 |
| 2 | `monotonic_queue.py` | 单调队列 | 队尾维护单调、队头过期弹出 | 摊还 O(1)/次 → 总 O(n) | LC 239 滑动窗口最大 |

## 选型

| 场景 | 选 | 备注 |
|------|-----|------|
| 固定容量 FIFO | Circular Queue | 内核/音频环形缓冲 |
| 窗口内 max/min | Monotonic Deque | 比堆删过期更干净 |
| 双端都能进 | `collections.deque` / `ArrayDeque` | 语言自带 |

## 面试 checklist

1. 环队列满空区分：size 计数，或浪费一格，或 flag。
2. 单调队列存「下标」还是「值」：要过期判断通常存下标。
3. LC 239：窗口从左滑右，维护递减双端队列。

## 自测

```bash
cd AdvancedQueue && for f in *.py; do python3 "$f"; done
```
