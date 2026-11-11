"""
AdvancedDP / DP 优化（单调队列 / 斜率优化骨架 / Knuth 条件说明）
把 O(n²) 或 O(n³) 转移加速到 O(n)/O(n²)
App: 滑动窗口最大分、划分优化、凸壳决策
"""

from collections import deque


def max_result_jump(nums, k):
    """每次跳 1..k，得分最大。LC 1696。单调队列优化。"""
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    q = deque([0])  # 下标，dp 递减
    for i in range(1, n):
        while q and q[0] < i - k:
            q.popleft()
        dp[i] = nums[i] + dp[q[0]]
        while q and dp[q[-1]] <= dp[i]:
            q.pop()
        q.append(i)
    return dp[-1]


def partition_min_cost_mono(cost_prefix, k_max=None):
    """示意：dp[i]=min_{j<i} dp[j]+w(j,i)，若 w 满足四边形可用更优；
    这里用朴素 O(n²) 对照，并给单调队列适用场景注释。
    cost_prefix 前缀和，w(j,i)=(pref[i]-pref[j])^2 凸。
    """
    n = len(cost_prefix) - 1
    dp = [0] + [float("inf")] * n
    for i in range(1, n + 1):
        for j in range(i):
            w = (cost_prefix[i] - cost_prefix[j]) ** 2
            dp[i] = min(dp[i], dp[j] + w)
    return dp[n]


def slope_optimize_partition(pref):
    """斜率优化划分：dp[i]=min_j dp[j] + (pref[i]-pref[j])^2
    = pref[i]^2 + min_j (dp[j]+pref[j]^2 - 2*pref[i]*pref[j])
    对 j 维护下凸壳，按 pref[i] 递增查询。
    """
    n = len(pref) - 1
    dp = [0] * (n + 1)
    # hull of lines: m = -2*pref[j], b = dp[j]+pref[j]^2 ；存 (m,b,j)
    hull = []  # (m, b)

    def bad(l1, l2, l3):
        # 斜率比较交叉：去掉中间
        return (l2[1] - l1[1]) * (l3[0] - l2[0]) >= (l3[1] - l2[1]) * (l2[0] - l1[0])

    def add_line(m, b):
        hull.append((m, b))
        while len(hull) >= 3 and bad(hull[-3], hull[-2], hull[-1]):
            hull.pop(-2)

    def query(x):
        # 查询 min m*x+b；m 递增（-2*pref 递减? pref 增 → m=-2p 递减）
        # 用双指针：本处简单线性扫（教学）；生产可用指针单调前移
        return min(m * x + b for m, b in hull)

    add_line(0, 0)  # j=0: pref0=0, dp0=0
    for i in range(1, n + 1):
        dp[i] = pref[i] * pref[i] + query(pref[i])
        add_line(-2 * pref[i], dp[i] + pref[i] * pref[i])
    return dp[n]


def knuth_opt_matrix_chain(dims):
    """Knuth 优化矩阵链：满足四边形不等式时，最优分割点单调。
    将 O(n³) 常数改善（仍 O(n²) 层，k 范围缩小）。
    """
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
            start = opt[i][j - 1]
            end = opt[i + 1][j] if i + 1 <= j else j - 1
            start = max(i, start)
            end = min(j - 1, end)
            for k in range(start, end + 1):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k
    return dp[0][n - 1]


def _test():
    assert max_result_jump([1, -1, -2, 4, -7, 3], 2) == 7
    pref = [0, 1, 3, 6, 10]  # stones 1,2,3,4
    naive = partition_min_cost_mono(pref)
    opt = slope_optimize_partition(pref)
    assert naive == opt
    assert knuth_opt_matrix_chain([10, 30, 5, 60]) == 4500
    print("dp_optimize OK")


APPLICATION = """
- 单调队列: 转移只依赖「窗口内最优 dp[j]」。
- 斜率优化: 决策代价可写成 m(j)*x(i)+b(j)，维护凸壳。
- Knuth: 四边形不等式 + 单调性 → 最优 k 区间收缩。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
