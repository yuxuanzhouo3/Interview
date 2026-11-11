"""
AdvancedDP / 线性·状态机 DP
按位置推进；状态表示阶段（持有/冷却/完成第 k 笔…）
App: LC 121/122/123/188 / 309；覆盖/跳跃思想
"""


def max_profit_k_transactions(k, prices):
    """至多 k 笔交易。LC 188。k 大时退化为无限次。"""
    n = len(prices)
    if n < 2 or k == 0:
        return 0
    if k >= n // 2:
        return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, n))
    # hold[t] / cash[t] after at most t completed sells
    hold = [float("-inf")] * (k + 1)
    cash = [0] * (k + 1)
    for p in prices:
        for t in range(k, 0, -1):
            cash[t] = max(cash[t], hold[t] + p)
            hold[t] = max(hold[t], cash[t - 1] - p)
    return cash[k]


def max_profit_cooldown(prices):
    """卖出后冷冻一天。LC 309。"""
    hold = float("-inf")
    cash = cool = 0
    for p in prices:
        prev_cash = cash
        cash = max(cash, cool)
        cool = hold + p
        hold = max(hold, prev_cash - p)
    return max(cash, cool)


def max_profit_two(prices):
    """至多两笔。LC 123。"""
    return max_profit_k_transactions(2, prices)


def house_robber(nums):
    """线性打家劫舍：相邻不偷。状态机 take/skip。"""
    take = skip = 0
    for x in nums:
        take, skip = skip + x, max(skip, take)
    return max(take, skip)


def jump_min_steps(nums):
    """跳跃游戏 II：贪心线性扫描（状态机：当前段最远）。LC 45。"""
    n = len(nums)
    steps = end = farthest = 0
    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])
        if i == end:
            steps += 1
            end = farthest
    return steps


def _test():
    assert max_profit_k_transactions(2, [3, 2, 6, 5, 0, 3]) == 7
    assert max_profit_two([3, 3, 5, 0, 0, 3, 1, 4]) == 6
    assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3
    assert house_robber([2, 7, 9, 3, 1]) == 12
    assert jump_min_steps([2, 3, 1, 1, 4]) == 2
    print("linear_state_dp OK")


APPLICATION = """
- 股票：hold/cash 数组按交易次数滚动；注意买入扣钱顺序。
- 冷冻期：多一个 cool 状态接住昨天卖出。
- 线性 Hard 常把「阶段」显式成状态机，避免二维糊在一起。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
