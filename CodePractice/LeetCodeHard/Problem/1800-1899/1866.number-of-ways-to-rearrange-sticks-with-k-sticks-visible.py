"""
LC 1866. Number of Ways to Rearrange Sticks With K Sticks Visible — Hard
https://leetcode.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/
Best: DP dp[i][j]=dp[i-1][j-1]+(i-1)*dp[i-1][j] — O(nk)
"""


class Solution:
    def rearrangeSticks(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j] * (i - 1)) % MOD
        return dp[n][k]


def _test():
    s = Solution()
    assert s.rearrangeSticks(3, 2) == 3
    assert s.rearrangeSticks(5, 5) == 1
    assert s.rearrangeSticks(20, 11) == 647427950
    print("LC 1866 OK")


if __name__ == "__main__":
    _test()

