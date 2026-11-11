"""
AdvancedDP / 树形 DP（选不选 / 路径最大 / 树上背包）
模板: 后序 DFS，先算子树再合并到父
App: LC 337 / 124 / 968 思想
来源: CodePractice/dp.py §树形
"""

from collections import defaultdict


def rob_tree(root):
    """打家劫舍 III：父子不能同时偷。LC 337。
    返回 (不偷 root 最大, 偷 root 最大)。root 为三元组 (val, left, right) 或 None。
    """

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
    """二叉树最大路径和（任意节点为拐点）。LC 124。"""
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


def tree_knapsack(tree, root, volumes, values, capacity):
    """树上背包：选子树物品，体积不超过 capacity。
    tree: parent->children 邻接表；volumes/values 按节点编号。
    """
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


def independent_set_weight(tree, root, weight):
    """树上最大权独立集：dp[u][0/1]。"""
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


def _test():
    #     3
    #    / \
    #   2   3
    #    \   \
    #     3   1
    t = (3, (2, None, (3, None, None)), (3, None, (1, None, None)))
    assert rob_tree(t) == 7
    assert max_path_sum((-10, (9, None, None), (20, (15, None, None), (7, None, None)))) == 42

    tree = defaultdict(list)
    tree[0] = [1, 2]
    vols = [0, 2, 3]
    vals = [0, 4, 5]
    assert tree_knapsack(tree, 0, vols, vals, 5) == 9

    undirected = defaultdict(list)
    for u, v in [(0, 1), (0, 2)]:
        undirected[u].append(v)
        undirected[v].append(u)
    assert independent_set_weight(undirected, 0, [1, 2, 3]) == 5
    print("tree_dp OK")


APPLICATION = """
- 选/不选：返回二元组避免全局数组打架。
- 路径题：子树贡献可「拐弯」更新全局，返回给父的只能单链。
- 树上背包：枚举子树体积，父容量逆序合并。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
