"""
AdvancedDP / 划分型 DP（切割 / 分段）
状态: dp[i] = 前缀 s[:i] 的最优切法
App: LC 139 / 132 / 1278 思想
来源: CodePractice/dp.py §划分
"""


def word_break(s, word_dict):
    """能否拆成词典单词。LC 139。"""
    words = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]


def word_break_ways(s, word_dict):
    """拆法种数（计数划分）。"""
    words = set(word_dict)
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] += dp[j]
    return dp[n]


def min_cut_palindrome(s):
    """最少回文分割次数。LC 132。"""
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 1 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True
    dp = [0] * n
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
            continue
        dp[i] = i
        for j in range(1, i + 1):
            if is_pal[j][i]:
                dp[i] = min(dp[i], dp[j - 1] + 1)
    return dp[n - 1] if n else 0


def palindrome_partition_all(s):
    """返回所有回文分割方案。LC 131。"""
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 1 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True
    res, path = [], []

    def dfs(start):
        if start == n:
            res.append(path[:])
            return
        for end in range(start, n):
            if is_pal[start][end]:
                path.append(s[start : end + 1])
                dfs(end + 1)
                path.pop()

    dfs(0)
    return res


def _test():
    assert word_break("leetcode", ["leet", "code"]) is True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
    assert word_break_ways("catsanddog", ["cat", "cats", "and", "sand", "dog"]) == 2
    assert min_cut_palindrome("aab") == 1
    assert min_cut_palindrome("a") == 0
    parts = palindrome_partition_all("aab")
    assert sorted(parts) == sorted([["a", "a", "b"], ["aa", "b"]])
    print("partition_dp OK")


APPLICATION = """
- 先预处理回文表 is_pal[i][j]，再 O(n²) 切分。
- Word Break：dp[j] 真且 s[j:i] 在词典 → dp[i]。
- Hard：代价版回文更改次数可嵌进划分转移。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
