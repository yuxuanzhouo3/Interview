# AdvancedWindow · Syllabus

对照：`CodePractice/window.py`、滑窗题谱、`Monotonic.py`。Hard 关键在 **可变窗约束、双滑窗差分、前缀和 + 单调队列**。

## 学习路径

```
无重复 / 至多 K 种 → 最小覆盖窗 → 替换容错
  → 固定窗 → 恰好 K（双窗差分）
  → 前缀和+单调队列 → 滑窗最值/最值差
```

## 文件大纲

| # | 文件 | 主题 | 必会点 | 复杂度直觉 | 典型题 |
|---|------|------|--------|------------|--------|
| 1 | `no_repeat_window.py` | 最长无重复子串 | 右扩左缩；上次下标 | O(n) | 3 |
| 2 | `at_most_k_distinct.py` | 至多 K 种字符/整数 | `Counter` + `len(map)` | O(n) | 340 / 159 |
| 3 | `min_cover_window.py` | 最小覆盖子串 | `need/valid` 收缩 | O(n) | 76 |
| 4 | `char_replace_window.py` | 替换容错最长重复 | `(窗长-maxfreq)≤k` | O(n) | 424 / 1004 |
| 5 | `fixed_window.py` | 固定长度窗 | 进一出一维护统计 | O(n) | 1456 / 567 |
| 6 | `exact_k_window.py` | 恰好 K 种 | `atMost(K)-atMost(K-1)` | O(n) | 992 |
| 7 | `prefix_mono_deque.py` | 前缀和 + 单调队列 | 最短/最长满足和约束 | O(n) | 862 / 209 |
| 8 | `sliding_window_max.py` | 滑窗最值 | 递减单调队列下标 | O(n) | 239 |
| 9 | `abs_diff_limit.py` | 窗内 max-min≤limit | 双单调队列 | O(n) | 1438 |
| 10 | `hard_window_apps.py` | 串联词 / 平衡替换等 | 步长跳跃或反转约束 | O(n·\|W\|) 等 | 30 / 1234 / 1358 |

## 模板怎么选

| 题感 | 选 |
|------|----|
| 最长满足「至多…」 | 可变右指针 + while 违规左缩 |
| 最短覆盖 | 先扩到合法再尽量缩 |
| 「恰好 K」 | 两个 atMost 相减 |
| 子数组和 ≥K 且最短 | 前缀和 + 单调队列 |
| 窗内最值 / 极差 | 单调队列 |

## 面试口述 checklist

1. 滑窗本质：维护区间可加信息，均摊每端点移动 O(1)。
2. 单调队列存下标，队头出界、队尾不单调则弹。
3. 字符串频次类用数组 `[26]`/`[128]` 常比哈希更快、好写。
4. Hard 862：前缀递增队列找最小 `j-i`。
5. 与单调栈分工：栈偏「一侧最近更大/更小」；队列偏「窗口内最值」。

## 自测

```bash
cd AdvancedWindow && for f in *.py; do python3 "$f"; done
```
