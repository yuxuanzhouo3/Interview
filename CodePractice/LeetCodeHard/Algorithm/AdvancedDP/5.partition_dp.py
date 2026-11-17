"""
5. 划分型 DP · 判定 / 最优划分 / 约束划分个数
对照: 灵茶山 DP 题单 §五
App: LC 139 / 132 / 91 / 410 / 1043 / 1278
合并自: partition_dp.py
状态: f[i]=前缀 a[:i]；枚举最后一段左端点 L，f[L]→f[i]
"""


def word_break(s, word_dict):
    """能否拆成词典单词。LC 139。§5.1 判定。"""
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
    """拆法种数。"""
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
    """最少回文分割次数。LC 132。§5.2 最优划分。"""
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
    """所有回文分割方案。LC 131。"""
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


def num_decodings(s):
    """解码方法。LC 91。"""
    if not s or s[0] == "0":
        return 0
    n = len(s)
    a, b = 1, 1  # dp[i-2], dp[i-1]
    for i in range(1, n):
        cur = 0
        if s[i] != "0":
            cur += b
        two = int(s[i - 1 : i + 1])
        if 10 <= two <= 26:
            cur += a
        a, b = b, cur
    return b


def split_array_largest_sum(nums, k):
    """分割数组的最大值（最小化最大段和）。LC 410。§5.3 + 二分。"""
    lo, hi = max(nums), sum(nums)

    def ok(mid):
        cnt = 1
        cur = 0
        for x in nums:
            if cur + x > mid:
                cnt += 1
                cur = 0
            cur += x
        return cnt <= k

    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _test():
    assert word_break("leetcode", ["leet", "code"]) is True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
    assert word_break_ways("catsanddog", ["cat", "cats", "and", "sand", "dog"]) == 2
    assert min_cut_palindrome("aab") == 1
    parts = palindrome_partition_all("aab")
    assert sorted(parts) == sorted([["a", "a", "b"], ["aa", "b"]])
    assert num_decodings("226") == 3
    assert split_array_largest_sum([7, 2, 5, 10, 8], 2) == 18
    print("5.partition_dp OK")


APPLICATION = """
- 判定: f[i]=能否；最优: 最少/最多段或代价；约束 k 段: f[i][j]。
- 枚举最后一段 [L,i)，从 f[L] 转移。
- 恰好 k 段可控制循环上下界，O(nk)→O((n-k)k)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
