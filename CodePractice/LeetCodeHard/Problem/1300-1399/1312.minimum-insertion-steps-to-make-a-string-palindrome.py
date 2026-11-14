"""
LC 1312. Minimum Insertion Steps to Make a String Palindrome — Hard
https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/
Best: n - LCS(s, reverse(s)) — O(n^2)
"""


class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        t = s[::-1]
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            prev = 0
            for j in range(1, n + 1):
                cur = dp[j]
                if s[i - 1] == t[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = cur
        return n - dp[n]


def _test():
    s = Solution()
    assert s.minInsertions("zzazz") == 0
    assert s.minInsertions("mbadm") == 2
    assert s.minInsertions("leetcode") == 5
    print("LC 1312 OK")


if __name__ == "__main__":
    _test()
