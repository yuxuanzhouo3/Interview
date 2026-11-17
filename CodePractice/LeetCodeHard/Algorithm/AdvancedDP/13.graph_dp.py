"""
13. 图 DP · 最短路计数 / K 站中转 / 拓扑序 DP
对照: 灵茶山 DP 题单 §十三
App: LC 787 / 1976 / 2050 / 1857 / 913
"""

from collections import deque
import heapq


def find_cheapest_price(n, flights, src, dst, k):
    """最多 k 次中转最便宜航班。LC 787。Bellman-Ford / 分层。"""
    INF = 10**18
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):
        nd = dist[:]
        for u, v, w in flights:
            if dist[u] < INF:
                nd[v] = min(nd[v], dist[u] + w)
        dist = nd
    return -1 if dist[dst] >= INF else dist[dst]


def count_paths(n, roads):
    """到达目的地的方案数（最短路条数）。LC 1976。Dijkstra + DP。"""
    MOD = 10**9 + 7
    g = [[] for _ in range(n)]
    for u, v, w in roads:
        g[u].append((v, w))
        g[v].append((u, w))
    INF = 10**18
    dist = [INF] * n
    ways = [0] * n
    dist[0] = 0
    ways[0] = 1
    pq = [(0, 0)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                ways[v] = ways[u]
                heapq.heappush(pq, (nd, v))
            elif nd == dist[v]:
                ways[v] = (ways[v] + ways[u]) % MOD
    return ways[n - 1]


def minimum_time_courses(n, relations, time):
    """并行课程 III。LC 2050。拓扑 DP。"""
    g = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in relations:
        g[a - 1].append(b - 1)
        indeg[b - 1] += 1
    finish = time[:]
    q = deque([i for i in range(n) if indeg[i] == 0])
    while q:
        u = q.popleft()
        for v in g[u]:
            finish[v] = max(finish[v], finish[u] + time[v])
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return max(finish)


def largest_path_value(colors, edges):
    """有向图最大颜色值。LC 1857。拓扑 + 计数 DP。"""
    n = len(colors)
    g = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        g[u].append(v)
        indeg[v] += 1
    cnt = [[0] * 26 for _ in range(n)]
    q = deque()
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)
    seen = 0
    ans = 0
    while q:
        u = q.popleft()
        seen += 1
        c = ord(colors[u]) - 97
        cnt[u][c] += 1
        ans = max(ans, cnt[u][c])
        for v in g[u]:
            for k in range(26):
                cnt[v][k] = max(cnt[v][k], cnt[u][k])
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return ans if seen == n else -1


def _test():
    assert find_cheapest_price(3, [[0, 1, 100], [1, 2, 100], [0, 2, 500]], 0, 2, 1) == 200
    assert count_paths(7, [[0, 6, 7], [0, 1, 2], [1, 2, 3], [1, 3, 3], [6, 3, 3], [3, 5, 1], [6, 5, 1], [2, 5, 1], [0, 4, 5], [4, 6, 2]]) == 4
    assert minimum_time_courses(3, [[1, 3], [2, 3]], [3, 2, 5]) == 8
    assert largest_path_value("abaca", [[0, 1], [0, 2], [2, 3], [3, 4]]) == 3
    print("13.graph_dp OK")


APPLICATION = """
- K 限制最短路: 分层 / 至多松弛 K+1 轮。
- 方案数: 最短路相等时累加 ways。
- 有向无环: 拓扑序刷 DP；有环返回特殊值。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
