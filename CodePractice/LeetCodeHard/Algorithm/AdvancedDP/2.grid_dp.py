"""
2. 网格图 DP · 路径数 / 最小路径 / 递增路径
对照: 灵茶山 DP 题单 §二
App: LC 62 / 63 / 64 / 120 / 329 / 741 / 1463
合并自: counting_dp.unique_paths / memo_dfs.longest_increasing_path / unique_paths_obstacles
"""

from functools import lru_cache


def unique_paths(m, n):
    """不同路径。LC 62。"""
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]


def unique_paths_with_obstacles(grid):
    """不同路径 II。LC 63。"""
    m, n = len(grid), len(grid[0])

    @lru_cache(None)
    def dfs(i, j):
        if i >= m or j >= n or grid[i][j] == 1:
            return 0
        if i == m - 1 and j == n - 1:
            return 1
        return dfs(i + 1, j) + dfs(i, j + 1)

    return dfs(0, 0)


def min_path_sum(grid):
    """最小路径和。LC 64。"""
    m, n = len(grid), len(grid[0])
    dp = [float("inf")] * (n + 1)
    dp[1] = 0
    for i in range(m):
        for j in range(n):
            dp[j + 1] = grid[i][j] + min(dp[j], dp[j + 1])
    return dp[n]


def minimum_total_triangle(triangle):
    """三角形最小路径和。LC 120。自底向上。"""
    dp = triangle[-1][:]
    for r in range(len(triangle) - 2, -1, -1):
        for j in range(len(triangle[r])):
            dp[j] = triangle[r][j] + min(dp[j], dp[j + 1])
    return dp[0]


def longest_increasing_path(matrix):
    """矩阵最长递增路径。LC 329。记忆化。"""
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


def _test():
    assert unique_paths(3, 7) == 28
    assert unique_paths_with_obstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == 2
    assert min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7
    assert minimum_total_triangle([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]) == 11
    assert longest_increasing_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]) == 4
    print("2.grid_dp OK")


APPLICATION = """
- 网格 DP 是二维表上的「移动」；先形象再学背包/LCS。
- 只向右/下：一维滚动即可；四向递增：记忆化 DFS。
- 摘樱桃类：两人同走或折返，状态要加双方坐标（或对称压缩）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
