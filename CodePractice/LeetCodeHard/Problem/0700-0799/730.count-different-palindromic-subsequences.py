"""
LC 730. Count Different Palindromic Subsequences — Hard
Best: Interval DP by char ends — O(n^2)
"""

MOD = 10**9 + 7


class Solution:
    def countPalindromicSubsequences(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] != s[j]:
                    dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]) % MOD
                else:
                    left, right = i + 1, j - 1
                    while left <= right and s[left] != s[i]:
                        left += 1
                    while left <= right and s[right] != s[i]:
                        right -= 1
                    if left > right:
                        dp[i][j] = (dp[i + 1][j - 1] * 2 + 2) % MOD
                    elif left == right:
                        dp[i][j] = (dp[i + 1][j - 1] * 2 + 1) % MOD
                    else:
                        dp[i][j] = (dp[i + 1][j - 1] * 2 - dp[left + 1][right - 1]) % MOD
        return dp[0][n - 1] % MOD


def _test():
    s = Solution()
    assert s.countPalindromicSubsequences("bccb") == 6
    print("LC 730 OK")


if __name__ == "__main__":
    _test()
