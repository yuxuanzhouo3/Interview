# AdvancedGraph · Syllabus

统一接口：`query` / `update` / `add` / `del`。建议按下方路径刷，先基础连通，再最短路，再结构分解，最后流与匹配。

## 学习路径

```
基础表示 → 连通/并查 → 拓扑/环 → 最短路族 → MST
    → 桥割点/SCC → 二分图 → 网络流 → 启发搜索
```

## 文件大纲

| # | 文件 | 主题 | 必会点 | 复杂度直觉 | 典型题/应用 |
|---|------|------|--------|------------|-------------|
| 1 | `adjacency.py` | 邻接表图 | CRUD 边/点、BFS | V+E | 社交图、依赖图建模 |
| 2 | `union_find.py` | 并查集 DSU | find 路径压缩、union by rank | α(n) | Kruskal、岛屿、账号合并 |
| 3 | `topo_sort.py` | 拓扑排序 Kahn | 入度队列、环则失败 | V+E | LC 210 课程表、CI 依赖 |
| 4 | `cycle.py` | 有向/无向环 | 三色 DFS / DSU | V+E | LC 207 / 684、死锁 |
| 5 | `dag_dp.py` | DAG DP | 拓扑序松弛最长/最短 | V+E | LC 329 思想、关键路径 |
| 6 | `shortest_path.py` | Dijkstra | 堆优化非负权 | (V+E) log V | LC 743 网络延时 |
| 7 | `bellman_ford.py` | Bellman-Ford | 松弛 V-1、第 V 轮负环 | VE | 负权、差分约束、套利 |
| 8 | `zero_one_bfs.py` | 0-1 BFS | deque：0 权前端 / 1 权后端 | V+E | 边权仅 0/1 的网格 |
| 9 | `floyd_warshall.py` | Floyd 全源 | `d[i][j]=min(..., i-k-j)` | V³ | 小稠密图、传递闭包 |
| 10 | `astar.py` | A* | f=g+h，可采纳启发 | 视启发 | 游戏寻路、网格 |
| 11 | `mst.py` | Kruskal / Prim | 边排序+DSU / 堆扩点 | E log E | LC 1135 / 1584 |
| 12 | `bridge_cut.py` | 桥 / 割点 | Tarjan dfn/low | V+E | LC 1192 关键连接 |
| 13 | `scc.py` | 强连通分量 | Tarjan / Kosaraju | V+E | 缩点、2-SAT |
| 14 | `bipartite.py` | 二分图 | 2-coloring、最大匹配 | V+E / 增广 | LC 785 / 886、任务分配 |
| 15 | `network_flow.py` | 最大流 Dinic | level graph + DFS 堵 | 较优实用 | 最小割、吞吐、匹配建流 |
| 16 | `grid_graph.py` | 网格图 | 四联通 BFS、连通块 | R·C | LC 200 岛屿；超市/迷宫网格 |

## 最短路怎么选

| 场景 | 选 |
|------|----|
| 非负权单源 | Dijkstra |
| 有负权 / 差分约束 | Bellman-Ford |
| 边权只有 0/1 | 0-1 BFS |
| 全源且 V 小 | Floyd |
| 单源单汇 + 启发 | A* |
| DAG | `dag_dp` / 拓扑松弛 |

## 面试口述 checklist

1. 并查集 vs DFS 连通：在线合并用 DSU，离线遍历用 DFS/BFS。
2. Dijkstra 不能有负边；负环用 BF 第 n 轮检测。
3. 桥：`low[v] > dfn[u]`；割点：根多子 / 非根 `low[v] >= dfn[u]`。
4. 二分图最大匹配 ≈ 网络流特化；一般最大流用 Dinic。
5. MST Kruskal 稀疏边方便；Prim 稠密点扩更自然。

## 自测

```bash
cd AdvancedGraph && for f in *.py; do python3 "$f"; done
```
