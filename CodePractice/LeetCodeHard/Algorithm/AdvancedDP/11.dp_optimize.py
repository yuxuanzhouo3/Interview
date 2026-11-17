"""
11. 优化 DP · 前缀和 / 单调队列 / 斜率 / Knuth / 矩阵快速幂思想
对照: 灵茶山 DP 题单 §十一
App: LC 1696 / 1425 / 2407 / 70 / 935
合并自: dp_optimize.py
"""

from collections import deque


def max_result_jump(nums, k):
    """每次跳 1..k 最大得分。LC 1696。单调队列。"""
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    q = deque([0])
    for i in range(1, n):
        while q and q[0] < i - k:
            q.popleft()
        dp[i] = nums[i] + dp[q[0]]
        while q and dp[q[-1]] <= dp[i]:
            q.pop()
        q.append(i)
    return dp[-1]


def constrained_subsequence_sum(nums, k):
    """带限制的子序列和。LC 1425。单调队列。"""
    n = len(nums)
    dp = nums[:]
    q = deque([0])
    ans = nums[0]
    for i in range(1, n):
        while q and q[0] < i - k:
            q.popleft()
        dp[i] = max(nums[i], nums[i] + dp[q[0]])
        ans = max(ans, dp[i])
        while q and dp[q[-1]] <= dp[i]:
            q.pop()
        q.append(i)
    return ans


def partition_min_cost_mono(cost_prefix, k_max=None):
    """朴素对照: dp[i]=min_j dp[j]+(pref[i]-pref[j])^2。"""
    n = len(cost_prefix) - 1
    dp = [0] + [float("inf")] * n
    for i in range(1, n + 1):
        for j in range(i):
            w = (cost_prefix[i] - cost_prefix[j]) ** 2
            dp[i] = min(dp[i], dp[j] + w)
    return dp[n]


def slope_optimize_partition(pref):
    """斜率优化划分（凸壳）。§11.7。"""
    n = len(pref) - 1
    dp = [0] * (n + 1)
    hull = []

    def bad(l1, l2, l3):
        return (l2[1] - l1[1]) * (l3[0] - l2[0]) >= (l3[1] - l2[1]) * (l2[0] - l1[0])

    def add_line(m, b):
        hull.append((m, b))
        while len(hull) >= 3 and bad(hull[-3], hull[-2], hull[-1]):
            hull.pop(-2)

    def query(x):
        return min(m * x + b for m, b in hull)

    add_line(0, 0)
    for i in range(1, n + 1):
        dp[i] = pref[i] * pref[i] + query(pref[i])
        add_line(-2 * pref[i], dp[i] + pref[i] * pref[i])
    return dp[n]


def knuth_opt_matrix_chain(dims):
    """Knuth 优化矩阵链。"""
    n = len(dims) - 1
    if n <= 1:
        return 0
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]
    for i in range(n):
        opt[i][i] = i
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            start = max(i, opt[i][j - 1])
            end = min(j - 1, opt[i + 1][j] if i + 1 <= j else j - 1)
            for k in range(start, end + 1):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k
    return dp[0][n - 1]


def mat_pow_fib(n):
    """矩阵快速幂求斐波那契 F(n)。§11.6 线性递推。F(0)=0,F(1)=1。"""
    if n == 0:
        return 0

    def mul(A, B):
        return [
            [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
        ]

    def power(M, e):
        R = [[1, 0], [0, 1]]
        while e:
            if e & 1:
                R = mul(R, M)
            M = mul(M, M)
            e >>= 1
        return R

    return power([[1, 1], [1, 0]], n)[0][1]


def _test():
    assert max_result_jump([1, -1, -2, 4, -7, 3], 2) == 7
    assert constrained_subsequence_sum([10, 2, -10, 5, 20], 2) == 37
    pref = [0, 1, 3, 6, 10]
    assert partition_min_cost_mono(pref) == slope_optimize_partition(pref)
    assert knuth_opt_matrix_chain([10, 30, 5, 60]) == 4500
    assert mat_pow_fib(10) == 55
    print("11.dp_optimize OK")


APPLICATION = """
- 单调队列: 窗口内最优转移；右端增则左端也增。
- 斜率/CHT: 代价写成 m(j)*x(i)+b(j)，维护凸壳。
- 矩阵快速幂: 常系数线性递推 O(k³ log n)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
