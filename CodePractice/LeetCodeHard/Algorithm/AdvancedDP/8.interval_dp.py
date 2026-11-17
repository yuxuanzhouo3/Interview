"""
8. 区间 DP · 回文子序列 / 合并 / 戳气球 / 三角剖分
对照: 灵茶山 DP 题单 §八
App: LC 516 / 5 / 312 / 1000 / 1039 / 664 / 96 / 87
合并自: interval_dp.py + subsequence_dp.longest_palindrome_subseq + counting_dp.num_trees + memo_dfs.is_scramble
模板: f[i][j]；枚举 length → left → 分割点 k
"""

from functools import lru_cache


def longest_palindrome_subseq(s):
    """最长回文子序列。LC 516。§8.1。"""
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = 2 if length == 2 else dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1] if n else 0


def burst_balloons(nums):
    """戳气球最大硬币。LC 312。开区间最后戳 k。"""
    a = [1] + list(nums) + [1]
    n = len(a)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            for k in range(left + 1, right):
                dp[left][right] = max(
                    dp[left][right],
                    dp[left][k] + dp[k][right] + a[left] * a[k] * a[right],
                )
    return dp[0][n - 1]


def matrix_chain(dims):
    """矩阵链乘最少乘法。"""
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n - 1] if n else 0


def merge_stones_cost(stones):
    """相邻两堆合并最少代价。"""
    n = len(stones)
    if n <= 1:
        return 0
    pref = [0] * (n + 1)
    for i, x in enumerate(stones):
        pref[i + 1] = pref[i] + x
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            s = pref[j + 1] - pref[i]
            for k in range(i, j):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + s)
    return dp[0][n - 1]


def min_score_triangulation(values):
    """多边形三角剖分最小分。LC 1039。"""
    n = len(values)
    dp = [[0] * n for _ in range(n)]
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            for k in range(i + 1, j):
                dp[i][j] = min(
                    dp[i][j],
                    dp[i][k] + dp[k][j] + values[i] * values[k] * values[j],
                )
    return dp[0][n - 1]


def num_trees(n):
    """不同 BST 个数 = Catalan。LC 96。区间思想。"""
    dp = [0] * (n + 1)
    dp[0] = 1
    for nodes in range(1, n + 1):
        for left in range(nodes):
            dp[nodes] += dp[left] * dp[nodes - 1 - left]
    return dp[n]


def is_scramble(s1, s2):
    """扰乱字符串。LC 87。区间 + 记忆化。"""

    @lru_cache(None)
    def dfs(a, b):
        if a == b:
            return True
        if sorted(a) != sorted(b):
            return False
        n = len(a)
        for i in range(1, n):
            if dfs(a[:i], b[:i]) and dfs(a[i:], b[i:]):
                return True
            if dfs(a[:i], b[-i:]) and dfs(a[i:], b[:-i]):
                return True
        return False

    return dfs(s1, s2)


def _test():
    assert longest_palindrome_subseq("bbbab") == 4
    assert burst_balloons([3, 1, 5, 8]) == 167
    assert matrix_chain([10, 30, 5, 60]) == 4500
    assert merge_stones_cost([1, 2, 3, 4]) == 19
    assert min_score_triangulation([1, 2, 3]) == 6
    assert num_trees(3) == 5
    assert is_scramble("great", "rgeat") is True
    print("8.interval_dp OK")


APPLICATION = """
- 外层区间长度，保证子区间已算完。
- 消除/匹配类: 性质2缩两端，性质3枚举偶数步分割点。
- 博弈区间见 §14。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
