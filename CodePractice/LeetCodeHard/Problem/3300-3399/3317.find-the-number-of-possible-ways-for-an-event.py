"""
LC 3317. Find the Number of Possible Ways for an Event — Hard
https://leetcode.com/problems/find-the-number-of-possible-ways-for-an-event/
Best: 方法一：动态规划
"""

class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        mod = 10**9 + 7
        f = [[0] * (x + 1) for _ in range(n + 1)]
        f[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, x + 1):
                f[i][j] = (f[i - 1][j] * j + f[i - 1][j - 1] * (x - (j - 1))) % mod
        ans, p = 0, 1
        for j in range(1, x + 1):
            p = p * y % mod
            ans = (ans + f[n][j] * p) % mod
        return ans

def _test():
    s = Solution()
    assert s.numberOfWays(n=1, x=2, y=3) == 6
    assert s.numberOfWays(n=5, x=2, y=1) == 32
    assert s.numberOfWays(n=3, x=3, y=4) == 684
    print("LC 3317 OK")

if __name__ == "__main__":
    _test()
