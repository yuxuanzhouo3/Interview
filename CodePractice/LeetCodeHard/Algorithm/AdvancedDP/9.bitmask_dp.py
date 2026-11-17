"""
9. 状态压缩 DP · 排列 / TSP / 子集 / SOS
对照: 灵茶山 DP 题单 §九
App: LC 526 / 847 / 698 / 943 / 2305 / 2044
合并自: bitmask_dp.py
n≤20 → O(n·2^n)；子集枚举 → O(3^n)
"""


def tsp(dist):
    """旅行商最短回路。§9.3。"""
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
    """访问所有节点最短路。LC 847。"""
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
    """枚举 mask 非空子集。"""
    sub = mask
    out = []
    while sub:
        out.append(sub)
        sub = (sub - 1) & mask
    return out


def can_partition_k_subsets(nums, k):
    """分成 k 个等和子集。LC 698。"""
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
                break
            if nxt_sum == target:
                nxt_sum = 0
            dp[mask | (1 << i)] = nxt_sum
    return dp[(1 << n) - 1] == 0


def sos_dp_subset_count(a):
    """SOS DP：f[S]=a 中是 S 的子集的元素个数。§9.6。"""
    mx = max(a) if a else 0
    w = mx.bit_length()
    u = 1 << w
    f = [0] * u
    for x in a:
        f[x] += 1
    for i in range(w):
        bit = 1 << i
        s = 0
        while s < u:
            s |= bit
            f[s] += f[s ^ bit]
            s += 1
    return f


def count_max_or_subsets(nums):
    """按位或最大值的子集数。LC 2044。"""
    n = len(nums)
    max_or = 0
    for x in nums:
        max_or |= x
    ans = 0
    for mask in range(1, 1 << n):
        cur = 0
        for i in range(n):
            if mask >> i & 1:
                cur |= nums[i]
        if cur == max_or:
            ans += 1
    return ans


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
    f = sos_dp_subset_count([1, 2, 3])
    assert f[3] == 3  # 1,2,3 都是 {0,1} 的子集
    assert count_max_or_subsets([3, 1]) == 2
    print("9.bitmask_dp OK")


APPLICATION = """
- 排列型① f[S]；相邻相关再加 last i → O(n²·2ⁿ)。
- 子集枚举: sub=(sub-1)&mask，总 O(3ⁿ)。
- SOS: 按位刷表做高维前缀和。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
