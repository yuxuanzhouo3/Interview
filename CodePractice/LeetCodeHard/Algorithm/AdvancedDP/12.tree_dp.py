"""
12. 树形 DP · 直径 / 独立集 / 支配集 / 换根 / 树上背包
对照: 灵茶山 DP 题单 §十二
App: LC 543 / 337 / 124 / 968 / 834 / 310
合并自: tree_dp.py；含换根模板骨架
"""

from collections import defaultdict


def rob_tree(root):
    """打家劫舍 III / 最大独立集。LC 337。返回 (不偷, 偷)。"""

    def dfs(node):
        if not node:
            return 0, 0
        val, left, right = node
        ln, ly = dfs(left)
        rn, ry = dfs(right)
        not_take = max(ln, ly) + max(rn, ry)
        take = val + ln + rn
        return not_take, take

    return max(dfs(root))


def max_path_sum(root):
    """二叉树最大路径和。LC 124。直径类。"""
    ans = [float("-inf")]

    def dfs(node):
        if not node:
            return 0
        val, left, right = node
        l = max(0, dfs(left))
        r = max(0, dfs(right))
        ans[0] = max(ans[0], val + l + r)
        return val + max(l, r)

    dfs(root)
    return ans[0]


def tree_diameter_binary(root):
    """二叉树直径（边数）。LC 543。"""
    ans = [0]

    def dfs(node):
        if not node:
            return 0
        _, left, right = node
        l = dfs(left)
        r = dfs(right)
        ans[0] = max(ans[0], l + r)
        return 1 + max(l, r)

    dfs(root)
    return ans[0]


def independent_set_weight(tree, root, weight):
    """树上最大权独立集。"""
    n = len(weight)
    dp0 = [0] * n
    dp1 = [0] * n

    def dfs(u, parent):
        dp1[u] = weight[u]
        dp0[u] = 0
        for v in tree[u]:
            if v == parent:
                continue
            dfs(v, u)
            dp0[u] += max(dp0[v], dp1[v])
            dp1[u] += dp0[v]

    dfs(root, -1)
    return max(dp0[root], dp1[root])


def tree_knapsack(tree, root, volumes, values, capacity):
    """树上背包 O(nW²)。"""
    n = len(volumes)
    dp = [[0] * (capacity + 1) for _ in range(n)]

    def dfs(u):
        for v in range(volumes[u], capacity + 1):
            dp[u][v] = values[u]
        for child in tree[u]:
            dfs(child)
            for cap in range(capacity, -1, -1):
                best = dp[u][cap]
                for k in range(cap + 1):
                    best = max(best, dp[u][cap - k] + dp[child][k])
                dp[u][cap] = best

    dfs(root)
    return dp[root][capacity]


def sum_of_distances_in_tree(n, edges):
    """树中距离之和。LC 834。换根 DP。"""
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    size = [1] * n
    ans = [0] * n

    def dfs1(u, fa):
        for v in g[u]:
            if v == fa:
                continue
            dfs1(v, u)
            size[u] += size[v]
            ans[u] += ans[v] + size[v]

    def dfs2(u, fa):
        for v in g[u]:
            if v == fa:
                continue
            ans[v] = ans[u] - size[v] + (n - size[v])
            dfs2(v, u)

    dfs1(0, -1)
    dfs2(0, -1)
    return ans


def _test():
    t = (3, (2, None, (3, None, None)), (3, None, (1, None, None)))
    assert rob_tree(t) == 7
    assert max_path_sum((-10, (9, None, None), (20, (15, None, None), (7, None, None)))) == 42
    assert tree_diameter_binary((1, (2, (4, None, None), (5, None, None)), (3, None, None))) == 3

    tree = defaultdict(list)
    tree[0] = [1, 2]
    assert tree_knapsack(tree, 0, [0, 2, 3], [0, 4, 5], 5) == 9

    undirected = defaultdict(list)
    for u, v in [(0, 1), (0, 2)]:
        undirected[u].append(v)
        undirected[v].append(u)
    assert independent_set_weight(undirected, 0, [1, 2, 3]) == 5
    assert sum_of_distances_in_tree(6, [[0, 1], [0, 2], [2, 3], [2, 4], [2, 5]]) == [8, 12, 6, 10, 10, 10]
    print("12.tree_dp OK")


APPLICATION = """
- 后序合并子树；选/不选返回二元组。
- 路径可拐弯更新全局，返回父只能单链。
- 换根: 先算以 0 为根，再推到儿子（去掉子树 y，挂上剩余 F）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
