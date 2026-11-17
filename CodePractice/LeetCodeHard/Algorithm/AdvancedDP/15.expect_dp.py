"""
15. 概率 DP / 期望 DP
对照: 灵茶山 DP 题单 §十五
App: LC 688 / 837 / 808 / 1467
期望通常列方程或按拓扑/倒推；概率正向枚举转移
"""


def knight_probability(n, k, row, column):
    """骑士在棋盘上的概率。LC 688。"""
    dirs = ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))
    dp = [[0.0] * n for _ in range(n)]
    dp[row][column] = 1.0
    for _ in range(k):
        ndp = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if dp[i][j] == 0:
                    continue
                for di, dj in dirs:
                    x, y = i + di, j + dj
                    if 0 <= x < n and 0 <= y < n:
                        ndp[x][y] += dp[i][j] / 8.0
        dp = ndp
    return sum(sum(r) for r in dp)


def new21_game(n, k, maxPts):
    """新 21 点。LC 837。期望/概率 + 滑动窗口。"""
    if k == 0 or n >= k - 1 + maxPts:
        return 1.0
    dp = [0.0] * (n + 1)
    dp[0] = 1.0
    window = 1.0
    ans = 0.0
    for i in range(1, n + 1):
        dp[i] = window / maxPts
        if i < k:
            window += dp[i]
        else:
            ans += dp[i]
        if i - maxPts >= 0:
            window -= dp[i - maxPts]
    return ans


def soup_servings(n):
    """分汤。LC 808。期望概率倒推。"""
    if n >= 4800:
        return 1.0
    m = (n + 24) // 25
    from functools import lru_cache

    @lru_cache(None)
    def dfs(a, b):
        if a <= 0 and b <= 0:
            return 0.5
        if a <= 0:
            return 1.0
        if b <= 0:
            return 0.0
        return 0.25 * (
            dfs(a - 4, b)
            + dfs(a - 3, b - 1)
            + dfs(a - 2, b - 2)
            + dfs(a - 1, b - 3)
        )

    return dfs(m, m)


def _test():
    p = knight_probability(3, 2, 0, 0)
    assert abs(p - 0.0625) < 1e-9
    assert abs(new21_game(10, 1, 10) - 1.0) < 1e-9
    assert abs(new21_game(6, 1, 10) - 0.6) < 1e-9
    assert abs(soup_servings(50) - 0.625) < 1e-9
    print("15.expect_dp OK")


APPLICATION = """
- 概率: 枚举合法转移乘概率累加。
- 期望: E=1+Σp·E' 或倒推边界；注意吸收态。
- 大 n 截断/数学极限（如 808 n≥4800）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
