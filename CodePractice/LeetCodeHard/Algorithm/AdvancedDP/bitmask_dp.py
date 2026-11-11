"""
AdvancedDP / 状态压缩 DP（TSP / 子集枚举 / 划分）
状态: dp[mask][u] 或 dp[mask]
App: LC 847 / 691 / 698；n≤20
来源: CodePractice/dp.py §状压
"""


def tsp(dist):
    """旅行商：从 0 出发访问所有点最短回路。"""
    n = len(dist)
    INF = 10**18
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        for u in range(n):
            if not (mask >> u) & 1 or dp[mask][u] >= INF:
                continue
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                nxt = mask | (1 << v)
                dp[nxt][v] = min(dp[nxt][v], dp[mask][u] + dist[u][v])
    full = (1 << n) - 1
    return min(dp[full][u] + dist[u][0] for u in range(n))


def shortest_path_all_nodes(graph):
    """访问所有节点最短路（边权 1 BFS 状压）。LC 847。"""
    from collections import deque

    n = len(graph)
    target = (1 << n) - 1
    q = deque()
    seen = [[False] * (1 << n) for _ in range(n)]
    for i in range(n):
        q.append((i, 1 << i, 0))
        seen[i][1 << i] = True
    while q:
        u, mask, d = q.popleft()
        if mask == target:
            return d
        for v in graph[u]:
            nmask = mask | (1 << v)
            if not seen[v][nmask]:
                seen[v][nmask] = True
                q.append((v, nmask, d + 1))
    return -1


def enumerate_subsets(mask):
    """枚举 mask 的所有非空子集（含自身）。"""
    sub = mask
    out = []
    while sub:
        out.append(sub)
        sub = (sub - 1) & mask
    return out


def can_partition_k_subsets(nums, k):
    """能否分成 k 个等和子集。LC 698。"""
    total = sum(nums)
    if k <= 0 or total % k:
        return False
    target = total // k
    nums = sorted(nums, reverse=True)
    n = len(nums)
    if nums[0] > target:
        return False
    dp = [-1] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] < 0:
            continue
        for i, x in enumerate(nums):
            if mask & (1 << i):
                continue
            nxt_sum = dp[mask] + x
            if nxt_sum > target:
                break  # 已排序，更大只会更超
            if nxt_sum == target:
                nxt_sum = 0
            dp[mask | (1 << i)] = nxt_sum
    return dp[(1 << n) - 1] == 0


def _test():
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    assert tsp(dist) == 80
    assert shortest_path_all_nodes([[1, 2, 3], [0], [0], [0]]) == 4
    assert set(enumerate_subsets(0b1011)) == {0b1011, 0b1010, 0b1001, 0b1000, 0b0011, 0b0010, 0b0001}
    assert can_partition_k_subsets([4, 3, 2, 3, 5, 2, 1], 4) is True
    assert can_partition_k_subsets([1, 2, 3, 4], 3) is False
    print("bitmask_dp OK")


APPLICATION = """
- n≤20：2^n * poly(n) 可接受。
- 子集枚举: for sub=mask; sub; sub=(sub-1)&mask。
- 图上「访问集合」用 BFS/Dijkstra 扩 (node, mask)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
