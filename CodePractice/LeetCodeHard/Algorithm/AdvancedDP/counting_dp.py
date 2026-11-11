"""
AdvancedDP / 计数类 DP（路径 / 子序列 / Catalan）
求方案数；注意取模、组合 vs 排列外层循环顺序
App: LC 62 / 115 / 96 / 940
"""

MOD = 10**9 + 7


def unique_paths(m, n):
    """网格路径数。LC 62。"""
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]


def distinct_subsequences(s, t):
    """s 的子序列等于 t 的个数。LC 115。"""
    m, n = len(s), len(t)
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, m + 1):
        for j in range(n, 0, -1):
            if s[i - 1] == t[j - 1]:
                dp[j] += dp[j - 1]
    return dp[n]


def num_trees(n):
    """不同 BST 个数 = Catalan。LC 96。"""
    dp = [0] * (n + 1)
    dp[0] = 1
    for nodes in range(1, n + 1):
        for left in range(nodes):
            dp[nodes] += dp[left] * dp[nodes - 1 - left]
    return dp[n]


def distinct_subseq_ii(s):
    """不同空非空子序列个数。LC 940。"""
    # last[c] = 以 c 结尾的子序列个数累计贡献点
    dp = 0  # 当前不同子序列总数
    last = {}
    for ch in s:
        new = (dp + 1) % MOD  # 所有旧序列 +ch，再加单字符
        dp = (dp + new - last.get(ch, 0)) % MOD
        last[ch] = new
    return dp % MOD


def climb_stairs(n):
    """爬楼梯排列计数。LC 70。"""
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def _test():
    assert unique_paths(3, 7) == 28
    assert distinct_subsequences("rabbbit", "rabbit") == 3
    assert num_trees(3) == 5
    assert distinct_subseq_ii("abc") == 7
    assert distinct_subseq_ii("aba") == 6
    assert climb_stairs(5) == 8
    print("counting_dp OK")


APPLICATION = """
- 组合数背包外层物品；排列外层容量（或等价状态机）。
- Catalan: dp[n]=Σ dp[i]*dp[n-1-i]。
- 940 类「不同子序列」用 last 字符去重，防重复计数。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
