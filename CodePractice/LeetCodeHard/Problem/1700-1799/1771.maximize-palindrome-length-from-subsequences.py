"""
LC 1771. Maximize Palindrome Length From Subsequences — Hard
https://leetcode.com/problems/maximize-palindrome-length-from-subsequences/
Best: LPS DP on word1+word2 with cross constraint — O(n^2)
"""


class Solution:
    def longestPalindrome(self, word1: str, word2: str) -> int:
        s = word1 + word2
        n = len(s)
        m = len(word1)
        dp = [[0] * n for _ in range(n)]
        ans = 0
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2 if j - i > 1 else 2
                    if i < m <= j:
                        ans = max(ans, dp[i][j])
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return ans


def _test():
    s = Solution()
    assert s.longestPalindrome("cacb", "cbba") == 5
    assert s.longestPalindrome("ab", "ab") == 3
    assert s.longestPalindrome("aa", "bb") == 0
    print("LC 1771 OK")


if __name__ == "__main__":
    _test()
