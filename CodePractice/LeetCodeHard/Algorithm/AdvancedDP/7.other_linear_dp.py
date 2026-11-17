"""
7. 其他线性 DP · 一维 / 子数组 / 合法子序列 / 多维 / 计数
对照: 灵茶山 DP 题单 §七
App: LC 32 / 152 / 368 / 446 / 221 / 887 / 940
合并自: subsequence_dp.max_subarray 变体 + counting_dp + linear jump
"""

MOD = 10**9 + 7


def max_product_subarray(nums):
    """乘积最大子数组。LC 152。维护 max/min。"""
    ans = imax = imin = nums[0]
    for x in nums[1:]:
        candidates = (x, imax * x, imin * x)
        imax = max(candidates)
        imin = min(candidates)
        ans = max(ans, imax)
    return ans


def longest_valid_parentheses(s):
    """最长有效括号。LC 32。"""
    n = len(s)
    dp = [0] * n
    ans = 0
    for i in range(1, n):
        if s[i] != ")":
            continue
        if s[i - 1] == "(":
            dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
        else:
            j = i - dp[i - 1] - 1
            if j >= 0 and s[j] == "(":
                dp[i] = dp[i - 1] + 2 + (dp[j - 1] if j >= 1 else 0)
        ans = max(ans, dp[i])
    return ans


def largest_divisible_subset(nums):
    """最大整除子集。LC 368。LIS 变体 + 回溯方案。"""
    if not nums:
        return []
    a = sorted(nums)
    n = len(a)
    dp = [1] * n
    prev = [-1] * n
    best_i = 0
    for i in range(n):
        for j in range(i):
            if a[i] % a[j] == 0 and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
        if dp[i] > dp[best_i]:
            best_i = i
    path = []
    while best_i >= 0:
        path.append(a[best_i])
        best_i = prev[best_i]
    return path[::-1]


def number_of_arithmetic_slices_subseq(nums):
    """等差数列划分 II - 子序列。LC 446。"""
    from collections import defaultdict

    n = len(nums)
    dp = [defaultdict(int) for _ in range(n)]
    ans = 0
    for i in range(n):
        for j in range(i):
            d = nums[i] - nums[j]
            cnt = dp[j][d]
            dp[i][d] += cnt + 1
            ans += cnt
    return ans


def maximal_square(matrix):
    """最大正方形边长平方。LC 221。§7.5 子矩形。"""
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    side = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == "1":
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                side = max(side, dp[i][j])
    return side * side


def distinct_subseq_ii(s):
    """不同非空子序列个数。LC 940。§7.4/7.7。"""
    dp = 0
    last = {}
    for ch in s:
        new = (dp + 1) % MOD
        dp = (dp + new - last.get(ch, 0)) % MOD
        last[ch] = new
    return dp % MOD


def jump_min_steps(nums):
    """跳跃游戏 II。LC 45。专题跳跃。"""
    n = len(nums)
    steps = end = farthest = 0
    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])
        if i == end:
            steps += 1
            end = farthest
    return steps


def _test():
    assert max_product_subarray([2, 3, -2, 4]) == 6
    assert longest_valid_parentheses(")()())") == 4
    assert set(largest_divisible_subset([1, 2, 4, 8])) == {1, 2, 4, 8}
    assert number_of_arithmetic_slices_subseq([2, 4, 6, 8, 10]) == 7
    assert maximal_square([["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"], ["1", "0", "0", "1", "0"]]) == 4
    assert distinct_subseq_ii("abc") == 7
    assert jump_min_steps([2, 3, 1, 1, 4]) == 2
    print("7.other_linear_dp OK")


APPLICATION = """
- 一维: 前缀转移；子数组维护额外维度(正负积/删除一次)。
- 合法子序列: f[x] 以元素值结尾，值域大用哈希。
- 多维 Hard: 先定状态维，再写转移；方案数注意取模。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
