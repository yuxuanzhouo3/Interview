"""
LC 1216. Valid Palindrome III — Hard
https://leetcode.com/problems/valid-palindrome-iii/
Tags: String, DP
Best: n - LPS ≤ k
"""

class Solution:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        n = len(s)
        # dp[i][j] = LPS length in s[i:j+1]
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
        return n - dp[0][n - 1] <= k


def _test():
    sol = Solution()
    assert sol.isValidPalindrome("abcdeca", 2) is True
    assert sol.isValidPalindrome("abbababa", 1) is True
    print("LC 1216 OK")


if __name__ == "__main__":
    _test()

