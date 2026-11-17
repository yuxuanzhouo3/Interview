"""
LC 3277. Maximum XOR Score Subarray Queries — Hard
https://leetcode.com/problems/maximum-xor-score-subarray-queries/
Best: 方法一：动态规划
"""

class Solution:
    def maximumSubarrayXor(
        self, nums: list[int], queries: list[list[int]]
    ) -> list[int]:
        n = len(nums)
        f = [[0] * n for _ in range(n)]
        g = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            f[i][i] = g[i][i] = nums[i]
            for j in range(i + 1, n):
                f[i][j] = f[i][j - 1] ^ f[i + 1][j]
                g[i][j] = max(f[i][j], g[i][j - 1], g[i + 1][j])
        return [g[l][r] for l, r in queries]

def _test():
    s = Solution()
    assert s.maximumSubarrayXor(nums=[2, 8, 4, 32, 16, 1], queries=[[0, 2], [1, 4], [0, 5]]) == [12, 60, 60]
    assert s.maximumSubarrayXor(nums=[0, 7, 3, 2, 8, 5, 1], queries=[[0, 3], [1, 5], [2, 4], [2, 6], [5, 6]]) == [7, 14, 11, 14, 5]
    print("LC 3277 OK")

if __name__ == "__main__":
    _test()
