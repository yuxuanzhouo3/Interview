# AdvancedGreedy · Syllabus

从 `AdvancedOO` 迁出的 **贪心 & 区间调度**（原 Optimal/Ordering）。对照 `hard.py`。

## 学习路径

```
区间选择 → 区间覆盖 → 反悔堆 → 删 Digits → 加油站/跳跃 → Hard 堆贪心
```

## 文件大纲

| # | 文件 | 主题 | 典型题 |
|---|------|------|--------|
| 1 | `interval_select.py` | 最多不重叠 | 435 / 452 |
| 2 | `interval_cover.py` | 最少覆盖 / 跳跃 | 1326 / 45 |
| 3 | `course_heap_greedy.py` | 期限反悔堆 | 630 |
| 4 | `heap_reselect.py` | 堆增幅再选 | 1354 / 502 |
| 5 | `remove_k_digits.py` | 删 K 位最小 | 402 |
| 6 | `gas_jump_greedy.py` | 加油站 / 跳跃 | 134 / 55 / 45 |
| 7 | `schedule_two_city.py` | 代价差排序 | 1029 |
| 8 | `hard_greedy_apps.py` | 871 / 1834 | Hard 综合 |

## 自测

```bash
cd AdvancedGreedy && for f in *.py; do python3 "$f"; done
```
