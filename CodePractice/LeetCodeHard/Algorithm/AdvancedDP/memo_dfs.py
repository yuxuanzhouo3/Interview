"""
AdvancedDP / 记忆化搜索（DFS + Memo）
难写递推时的 Hard 首选；状态哈希或 @cache
App: LC 329 / 10 / 44 / 87 思想
"""

from functools import lru_cache


def longest_increasing_path(matrix):
    """矩阵最长递增路径。LC 329。"""
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    @lru_cache(None)
    def dfs(i, j):
        best = 1
        for di, dj in dirs:
            x, y = i + di, j + dj
            if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:
                best = max(best, 1 + dfs(x, y))
        return best

    return max(dfs(i, j) for i in range(m) for j in range(n))


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


def is_scramble(s1, s2):
    """扰乱字符串。LC 87。"""

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


def unique_paths_obstacles(grid):
    """带障碍不同路径（记忆化版）。"""
    m, n = len(grid), len(grid[0])

    @lru_cache(None)
    def dfs(i, j):
        if i >= m or j >= n or grid[i][j] == 1:
            return 0
        if i == m - 1 and j == n - 1:
            return 1
        return dfs(i + 1, j) + dfs(i, j + 1)

    return dfs(0, 0)


def _test():
    assert longest_increasing_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]) == 4
    assert is_match_regex("aab", "c*a*b") is True
    assert is_match_regex("mississippi", "mis*is*p*.") is False
    assert is_match_wildcard("adceb", "*a*b") is True
    assert is_match_wildcard("acdcb", "a*c?b") is False
    assert is_scramble("great", "rgeat") is True
    assert is_scramble("abcde", "caebd") is False
    assert unique_paths_obstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == 2
    print("memo_dfs OK")


APPLICATION = """
- 状态必须包含影响未来的全部信息；用 @cache 或 dict。
- 匹配类：指针 (i,j)；图路径类：(x,y) + 访问约束。
- 与递推等价：记忆化 = 按需填表，方便剪枝。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
