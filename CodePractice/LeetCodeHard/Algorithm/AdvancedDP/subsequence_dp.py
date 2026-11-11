"""
AdvancedDP / 子序列·子数组 DP（LIS / LCS / 回文子序列 / 编辑距离）
App: LC 300 / 1143 / 516 / 53 / 72
来源: CodePractice/dp.py §子序列
"""

import bisect


def lis_length(nums):
    """O(n²) LIS 长度。"""
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


def lcs(a, b):
    """最长公共子序列长度。LC 1143。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def longest_palindrome_subseq(s):
    """最长回文子序列。LC 516。"""
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


def max_subarray(nums):
    """Kadane。LC 53。"""
    cur = res = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        res = max(res, cur)
    return res


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


def _test():
    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lis_length_nlogn([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lcs("abcde", "ace") == 3
    assert longest_palindrome_subseq("bbbab") == 4
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert edit_distance("horse", "ros") == 3
    print("subsequence_dp OK")


APPLICATION = """
- LIS Hard 变体常叠二分 / 树状数组求「严格递增个数」。
- LCS / 编辑距离：对齐二维表；相等走对角，否则取三向最优。
- 回文子序列：两端相等 +2，否则舍一端。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
