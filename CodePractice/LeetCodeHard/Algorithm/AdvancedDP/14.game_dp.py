"""
14. 博弈 DP · 石子 / 预测赢家 / 先手最优
对照: 灵茶山 DP 题单 §十四
App: LC 486 / 877 / 1406 / 1140 / 1510 / 1025
常与区间 DP 结合: f[i][j]=区间上先手相对收益
"""


def predict_the_winner(nums):
    """预测赢家。LC 486。f=先手-后手差值。"""
    n = len(nums)
    dp = nums[:]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            # 新一轮: 选 nums[i] 则对手在 [i+1,j] 的差值取负
            dp[i] = max(nums[i] - dp[i + 1], nums[j] - dp[i])
    return dp[0] >= 0


def stone_game_iii(stone_value):
    """石子游戏 III。LC 1406。返回 'Alice'/'Bob'/'Tie'。"""
    n = len(stone_value)
    # dp[i] = 从 i 开始先手相对得分最优
    dp = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        best = float("-inf")
        take = 0
        for k in range(i, min(i + 3, n)):
            take += stone_value[k]
            best = max(best, take - dp[k + 1])
        dp[i] = best
    if dp[0] > 0:
        return "Alice"
    if dp[0] < 0:
        return "Bob"
    return "Tie"


def stone_game_ii(piles):
    """石子游戏 II。LC 1140。返回 Alice 石子数。"""
    n = len(piles)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + piles[i]
    from functools import lru_cache

    @lru_cache(None)
    def dfs(i, m):
        if i >= n:
            return 0
        if i + 2 * m >= n:
            return suffix[i]
        best = 0
        for x in range(1, 2 * m + 1):
            best = max(best, suffix[i] - dfs(i + x, max(m, x)))
        return best

    return dfs(0, 1)


def winner_square_game(n):
    """石子游戏 IV。LC 1510。先手能否赢。"""
    dp = [False] * (n + 1)
    for i in range(1, n + 1):
        x = 1
        while x * x <= i:
            if not dp[i - x * x]:
                dp[i] = True
                break
            x += 1
    return dp[n]


def divisor_game(n):
    """除数博弈。LC 1025。数学: n 偶先手赢。"""
    return n % 2 == 0


def _test():
    assert predict_the_winner([1, 5, 2]) is False
    assert predict_the_winner([1, 5, 233, 7]) is True
    assert stone_game_iii([1, 2, 3, 7]) == "Bob"
    assert stone_game_iii([1, 2, 3, -9]) == "Alice"
    assert stone_game_ii([2, 7, 9, 4, 4]) == 10
    assert winner_square_game(7) is False
    assert winner_square_game(1) is True
    assert divisor_game(2) is True
    print("14.game_dp OK")


APPLICATION = """
- 区间博弈: 选两端，对手最优 → 差值或绝对得分。
- 必胜态: 存在转移到必败态；必败态: 全是必胜转移。
- 部分石子题有数学解，DP 作通解。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
