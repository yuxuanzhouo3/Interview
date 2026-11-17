"""
4. 经典线性 DP · LCS / LIS / 编辑距离 / 子序列匹配
对照: 灵茶山 DP 题单 §四
App: LC 1143 / 72 / 300 / 115 / 10 / 44 / 97 / 1092
合并自: subsequence_dp + counting_dp.distinct_subsequences + memo_dfs 匹配
"""

import bisect
from functools import lru_cache


def lcs(a, b):
    """最长公共子序列。LC 1143。f[i][j] 对 (s[:i], t[:j])。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def edit_distance(a, b):
    """编辑距离。LC 72。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def lis_length(nums):
    """O(n²) LIS。LC 300。枚举选哪个。"""
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0


def lis_length_nlogn(nums):
    """O(n log n) 贪心 + 二分 tail。"""
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


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


def is_match_regex(s, p):
    """正则 '.' 与 '*'。LC 10。"""

    @lru_cache(None)
    def dfs(i, j):
        if j == len(p):
            return i == len(s)
        first = i < len(s) and (p[j] == s[i] or p[j] == ".")
        if j + 1 < len(p) and p[j + 1] == "*":
            return dfs(i, j + 2) or (first and dfs(i + 1, j))
        return first and dfs(i + 1, j + 1)

    return dfs(0, 0)


def is_match_wildcard(s, p):
    """通配 '?' 与 '*'。LC 44。"""

    @lru_cache(None)
    def dfs(i, j):
        if j == len(p):
            return i == len(s)
        if p[j] == "*":
            return dfs(i, j + 1) or (i < len(s) and dfs(i + 1, j))
        if i < len(s) and (p[j] == "?" or p[j] == s[i]):
            return dfs(i + 1, j + 1)
        return False

    return dfs(0, 0)


def is_interleave(s1, s2, s3):
    """交错字符串。LC 97。"""
    if len(s1) + len(s2) != len(s3):
        return False
    m, n = len(s1), len(s2)
    dp = [False] * (n + 1)
    dp[0] = True
    for j in range(1, n + 1):
        dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]
    for i in range(1, m + 1):
        dp[0] = dp[0] and s1[i - 1] == s3[i - 1]
        for j in range(1, n + 1):
            dp[j] = (dp[j] and s1[i - 1] == s3[i + j - 1]) or (
                dp[j - 1] and s2[j - 1] == s3[i + j - 1]
            )
    return dp[n]


def _test():
    assert lcs("abcde", "ace") == 3
    assert edit_distance("horse", "ros") == 3
    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lis_length_nlogn([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert distinct_subsequences("rabbbit", "rabbit") == 3
    assert is_match_regex("aab", "c*a*b") is True
    assert is_match_wildcard("adceb", "*a*b") is True
    assert is_interleave("aabcc", "dbbca", "aadbbcbcac") is True
    print("4.linear_classic_dp OK")


APPLICATION = """
- LCS/编辑距离：二维对齐；相等走对角，否则取相邻最优。
- LIS：枚举选哪个 / 二分 tail / 用 LCS 求 / 树状数组优化。
- 匹配类 Hard：优先记忆化 (i,j)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
