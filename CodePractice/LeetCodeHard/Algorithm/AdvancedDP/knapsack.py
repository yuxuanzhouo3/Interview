"""
AdvancedDP / Knapsack（0-1 / 完全 / 多重 / 多维）
状态: dp[w] = 容量 w 下最大价值（或方案数）
App: LC 416 / 322 / 518 / 474；能否凑成、最少件数、组合数。
来源: CodePractice/dp.py §背包
"""


def zero_one(weights, values, capacity):
    """每件至多一次；一维必须逆序。"""
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for cap in range(capacity, w - 1, -1):
            dp[cap] = max(dp[cap], dp[cap - w] + v)
    return dp[capacity]


def complete(weights, values, capacity):
    """每件无限次；一维正序。"""
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for cap in range(w, capacity + 1):
            dp[cap] = max(dp[cap], dp[cap - w] + v)
    return dp[capacity]


def multiple(weights, values, counts, capacity):
    """每件最多 count 次；二进制拆分后当 0-1。"""
    ws, vs = [], []
    for w, v, c in zip(weights, values, counts):
        k = 1
        while k <= c:
            ws.append(w * k)
            vs.append(v * k)
            c -= k
            k <<= 1
        if c:
            ws.append(w * c)
            vs.append(v * c)
    return zero_one(ws, vs, capacity)


def coin_change_min(coins, amount):
    """完全背包：凑 amount 最少硬币数。LC 322。"""
    INF = amount + 1
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] <= amount else -1


def coin_change_ways(coins, amount):
    """完全背包组合数（外层物品）。LC 518。"""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]


def can_partition(nums):
    """0-1：能否分割等和。LC 416。"""
    s = sum(nums)
    if s & 1:
        return False
    target = s // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for x in nums:
        for t in range(target, x - 1, -1):
            dp[t] = dp[t] or dp[t - x]
    return dp[target]


def ones_and_zeros(strs, m, n):
    """二维 0-1 背包。LC 474。"""
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for s in strs:
        zeros, ones = s.count("0"), s.count("1")
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    return dp[m][n]


def _test():
    assert zero_one([2, 3, 4], [3, 4, 5], 5) == 7  # 2+3
    assert complete([2, 3], [3, 4], 5) == 7  # 2+3 or 1*2+1*3
    assert multiple([2, 3], [3, 4], [2, 1], 5) == 7
    assert coin_change_min([1, 2, 5], 11) == 3
    assert coin_change_ways([1, 2, 5], 5) == 4
    assert can_partition([1, 5, 11, 5]) is True
    assert can_partition([1, 2, 3, 5]) is False
    assert ones_and_zeros(["10", "0001", "111001", "1", "0"], 5, 3) == 4
    print("knapsack OK")


APPLICATION = """
- 0-1 逆序防重复用；完全正序可累加同物。
- 求方案数外层物品、内层容量 → 组合；反过来常变排列。
- 多维容量（m个0、n个1）嵌套逆序滚动。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
