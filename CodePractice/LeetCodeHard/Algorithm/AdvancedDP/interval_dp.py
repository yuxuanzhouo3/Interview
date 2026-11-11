"""
AdvancedDP / 区间 DP（戳气球 / 矩阵链 / 石子合并）
模板: 枚举 length → left → 分割点 k
App: LC 312 / 1000 / 1039 / 664
来源: CodePractice/dp.py §区间
"""


def burst_balloons(nums):
    """戳气球最大硬币。LC 312。"""
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
    """矩阵链乘最少乘法。dims 长度 = 矩阵数+1。"""
    n = len(dims) - 1  # n matrices
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
    """相邻两堆合并最少代价（每次合 2 堆）。"""
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


def _test():
    assert burst_balloons([3, 1, 5, 8]) == 167
    assert matrix_chain([10, 30, 5, 60]) == 4500
    assert merge_stones_cost([1, 2, 3, 4]) == 19  # (1+2)=3 → (3+3)=6 → (6+4)=10
    assert min_score_triangulation([1, 2, 3]) == 6
    print("interval_dp OK")


APPLICATION = """
- 外层区间长度，保证转移时子区间已算完。
- 戳气球：开区间 (left,right) 最后戳 k。
- Hard 合并石子 K 堆：长度需满足 (len-1)%(K-1)==0。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
