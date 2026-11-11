# AdvancedClassifyThinking · Syllabus

Hard 题先**归类再动手**：把题面压成已知范式，再选 `Algorithm/` / `DataStructure/` 模板。

## 学习路径

```
题型路由 → 答案二分 / 折半搜索
  → 问题转化 / 离线 / 增量
  → 构造 / 不变量 → Hard 模式地图
```

## 文件大纲

| # | 文件 | 主题 | 必会点 | 典型题感 |
|---|------|------|--------|----------|
| 1 | `type_router.py` | 题型路由 | 关键词 → DP/图/窗/贪心/设计 | 开题 30 秒决策 |
| 2 | `answer_binary_search.py` | 答案二分 | 单调判定 `check(mid)` | 最小化最大 / 410 / 875 |
| 3 | `meet_in_middle.py` | 折半枚举 | n≤40 拆两半 | 子集和、2035 思想 |
| 4 | `reduction.py` | 问题转化 | 建图/建流/建差分约束 | Hard 建模 |
| 5 | `offline_query.py` | 离线思维 | 排序查询 + 扫 / 倒序加边 | 1697 / 动态连通 |
| 6 | `incremental.py` | 增量 / 差分 | 差分数组、事件扫描 | 扫线、区间加 |
| 7 | `constructive.py` | 构造题 | 先找可行结构再填 | 字典序/排列构造 |
| 8 | `hard_pattern_map.py` | Hard 模式地图 | 题感 → 模板文件索引 | 复习总表 |

## 开题决策树（口述）

1. 最优值且「猜答案可 O(n) 验证」→ 答案二分。
2. n≤40 子集/配对 → meet in middle 或状压。
3. 多条件连通/约束 → 并查集 / 网络流 / 差分约束。
4. 子数组约束滑动 → Window；局部最优可证 → OO 贪心。
5. 区间合并/划分 → 区间 DP；树 → 树形 DP。

## 自测

```bash
cd AdvancedClassifyThinking && for f in *.py; do python3 "$f"; done
```
