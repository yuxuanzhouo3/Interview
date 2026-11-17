"""
6. 状态机 DP · 买卖股票 / 阶段状态
对照: 灵茶山 DP 题单 §六
App: LC 121 / 122 / 123 / 188 / 309 / 714 / 198 / 376
合并自: linear_state_dp.py
状态: f[i][j] = 前缀 a[:i] 在状态 j 下的最优值
"""


def max_profit_one(prices):
    """交易 1 次。LC 121。"""
    buy = float("inf")
    ans = 0
    for p in prices:
        buy = min(buy, p)
        ans = max(ans, p - buy)
    return ans


def max_profit_unlimited(prices):
    """交易次数不限。LC 122。"""
    return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, len(prices)))


def max_profit_k_transactions(k, prices):
    """至多 k 笔。LC 188。k 大时退化为无限次。"""
    n = len(prices)
    if n < 2 or k == 0:
        return 0
    if k >= n // 2:
        return max_profit_unlimited(prices)
    hold = [float("-inf")] * (k + 1)
    cash = [0] * (k + 1)
    for p in prices:
        for t in range(k, 0, -1):
            cash[t] = max(cash[t], hold[t] + p)
            hold[t] = max(hold[t], cash[t - 1] - p)
    return cash[k]


def max_profit_two(prices):
    """至多两笔。LC 123。"""
    return max_profit_k_transactions(2, prices)


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


def max_profit_fee(prices, fee):
    """含手续费。LC 714。"""
    hold = float("-inf")
    cash = 0
    for p in prices:
        hold = max(hold, cash - p)
        cash = max(cash, hold + p - fee)
    return cash


def house_robber(nums):
    """打家劫舍也可看状态机 take/skip。LC 198。"""
    take = skip = 0
    for x in nums:
        take, skip = skip + x, max(skip, take)
    return max(take, skip)


def wiggle_max_length(nums):
    """摆动序列最长。LC 376。up/down 状态。"""
    if not nums:
        return 0
    up = down = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            up = down + 1
        elif nums[i] < nums[i - 1]:
            down = up + 1
    return max(up, down)


def _test():
    assert max_profit_one([7, 1, 5, 3, 6, 4]) == 5
    assert max_profit_unlimited([7, 1, 5, 3, 6, 4]) == 7
    assert max_profit_k_transactions(2, [3, 2, 6, 5, 0, 3]) == 7
    assert max_profit_two([3, 3, 5, 0, 0, 3, 1, 4]) == 6
    assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3
    assert max_profit_fee([1, 3, 2, 8, 4, 9], 2) == 8
    assert house_robber([2, 7, 9, 3, 1]) == 12
    assert wiggle_max_length([1, 7, 4, 9, 2, 5]) == 6
    print("6.state_machine_dp OK")


APPLICATION = """
- 股票: hold/cash 按交易次数滚动；注意买入扣钱顺序。
- 冷冻期多 cool；手续费在卖出扣。
- 线性 Hard 常把「阶段」显式成少量状态 j。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
