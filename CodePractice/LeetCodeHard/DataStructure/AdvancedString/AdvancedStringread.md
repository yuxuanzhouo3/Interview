# AdvancedString · Syllabus

统一接口：`query` / `update` / `add` / `del`。按匹配 → 回文 → 多模式 → 索引级结构推进。

## 学习路径

```
KMP → Z → Rabin-Karp / DoubleHash
  → Manacher
  → Aho-Corasick
  → Suffix Array → SAM
```

## 文件大纲

| # | 文件 | 算法 | 必会点 | 复杂度 | 典型应用 |
|---|------|------|--------|--------|----------|
| 1 | `kmp.py` | KMP | `pi[]` 最长真前后缀、失配跳转 | O(n+m) | LC 28 / 1392、周期串 |
| 2 | `z_algorithm.py` | Z 算法 | Z-box `[l,r]` 加速 | O(n) | 扩展 KMP、自身匹配 |
| 3 | `rabin_karp.py` | 滚动哈希 | 窗口进/出字符、模运算 | 均 O(n+m) | 子串相等、剽窃初筛 |
| 4 | `manacher.py` | Manacher | 插 `#`、半径数组 | O(n) | LC 5 最长回文 |
| 5 | `aho_corasick.py` | AC 自动机 | Trie + fail 指针 | O(n+命中+模式总长) | 敏感词、多关键字 |
| 6 | `suffix_array.py` | SA + LCP | 倍增求 SA、Kasai LCP | ~n log² n | 最长重复子串 LC 1044 |
| 7 | `suffix_automaton.py` | SAM | 后缀自动机、occ/本质不同子串 | O(n) 状态 | 竞赛字符串、基因组 |
| 8 | `string_hash_lcp.py` | 双哈希 LCP | 前缀哈希、二分 LCP | 预处理 O(n)，询 O(log n) | 快判等、字典序比较 |

## 怎么选

| 场景 | 选 |
|------|----|
| 单个模式精确匹配 | KMP / Z |
| 只要哈希判等 / 多模式初筛 | Rabin-Karp / DoubleHash |
| 最长回文子串 | Manacher |
| 很多模式同时搜 | Aho-Corasick |
| 最长重复 / 后缀相关 | Suffix Array |
| 子串出现次数、本质不同子串 | SAM |

## 面试 checklist

1. `pi[i]`：`s[0..i]` 最长 **proper** 前后缀；周期：`n % (n-pi[n-1])==0`。
2. RK 必须讲冲突：双模 / 再比原文。
3. AC 的 fail =「当前前缀的最长真后缀」在自动机上的跳转，是 KMP 的多模式推广。
4. SA `[i]` = 第 i 小后缀的起始下标；相邻后缀 LCP 用 Kasai。
5. 工程搜索多倒排 + FST；竞赛字符串重 SA/SAM/哈希。

## 自测

```bash
cd AdvancedString && for f in *.py; do python3 "$f"; done
```
