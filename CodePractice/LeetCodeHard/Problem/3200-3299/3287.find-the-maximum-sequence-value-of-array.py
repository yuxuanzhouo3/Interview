"""
LC 3287. Find the Maximum Sequence Value of Array — Hard
https://leetcode.com/problems/find-the-maximum-sequence-value-of-array/
Best: 方法一：动态规划 + 前后缀分解 + 枚举
"""

class Solution:
    def maxValue(self, nums: list[int], k: int) -> int:
        m = 1 << 7
        n = len(nums)
        f = [[[False] * m for _ in range(k + 2)] for _ in range(n + 1)]
        f[0][0][0] = True
        for i in range(n):
            for j in range(k + 1):
                for x in range(m):
                    f[i + 1][j][x] |= f[i][j][x]
                    f[i + 1][j + 1][x | nums[i]] |= f[i][j][x]

        g = [[[False] * m for _ in range(k + 2)] for _ in range(n + 1)]
        g[n][0][0] = True
        for i in range(n, 0, -1):
            for j in range(k + 1):
                for y in range(m):
                    g[i - 1][j][y] |= g[i][j][y]
                    g[i - 1][j + 1][y | nums[i - 1]] |= g[i][j][y]

        ans = 0
        for i in range(k, n - k + 1):
            for x in range(m):
                if f[i][k][x]:
                    for y in range(m):
                        if g[i][k][y]:
                            ans = max(ans, x ^ y)
        return ans

def _test():
    s = Solution()
    assert s.maxValue(nums=[2, 6, 7], k=1) == 5
    assert s.maxValue(nums=[4, 2, 5, 6, 7], k=2) == 2
    print("LC 3287 OK")

if __name__ == "__main__":
    _test()
