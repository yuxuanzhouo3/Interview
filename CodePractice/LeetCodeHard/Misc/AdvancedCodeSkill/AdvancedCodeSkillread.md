# AdvancedCodeSkill · Syllabus

实现层「手感」：二分/双指针/前缀差分/位运算/回溯剪枝/设计组合。与 `ClassifyThinking` 互补——那边选范式，这边写对边界。

## 学习路径

```
二分模板 → 双指针 → 前缀/差分
  → 位运算 / 快速幂 → 回溯剪枝
  → 单调结构 → 哈希+链表/堆设计 → Bloom
```

## 文件大纲

| # | 文件 | 主题 | 必会点 | 典型题 |
|---|------|------|--------|--------|
| 1 | `binary_search_template.py` | 上下界二分 | `lo/hi` 开闭；避免死循环 | 34 / 875 |
| 2 | `two_pointers.py` | 对撞 / 同向双指针 | 有序对、快慢指针 | 15 / 42 思想 |
| 3 | `prefix_diff.py` | 前缀和 + 差分 | 区间加 → 差分数组 | 370 / 1109 |
| 4 | `bit_ops.py` | 位运算技巧 | `i&-i`、子集枚举、位计数 | 191 / 78 思想 |
| 5 | `fast_pow_mod.py` | 快速幂 / 模运算 | 二进制拆分；防溢出 | 50 / 372 |
| 6 | `backtrack_prune.py` | 回溯 + 剪枝 | 序约束、可行性剪枝 | 51 / 37 |
| 7 | `lru_design.py` | 哈希 + 双向链表 | O(1) get/put | 146 |
| 8 | `bloom_filter.py` | Bloom Filter | 多哈希误判率 | 缓存穿透防护 |

## 编码 checklist

1. 二分先写 `check`，再决定答案落在 `lo` 还是 `hi`。
2. 差分：`d[l]+=v; d[r+1]-=v`，再前缀还原。
3. 回溯：改状态 → 递归 → **撤销**；剪枝写在扩展前。
4. LRU：HashMap + 双向链表；LFU 再加频次桶。
5. Bloom：只能「可能在 / 一定不在」，不能删（Counting Bloom 除外）。

## 自测

```bash
cd AdvancedCodeSkill && for f in *.py; do python3 "$f"; done
```
