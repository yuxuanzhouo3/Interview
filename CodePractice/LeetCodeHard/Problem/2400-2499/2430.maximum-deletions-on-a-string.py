"""
LC 2430. Maximum Deletions on a String — Hard
https://leetcode.com/problems/maximum-deletions-on-a-string/
Best: DP + Z-box / LCP equality for prefix deletes — O(n^2)
"""


class Solution:
    def deleteString(self, s: str) -> int:
        n = len(s)
        # lcp[i][j] = LCP of s[i:] and s[j:]
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == s[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1
        dp = [1] * n
        for i in range(n - 1, -1, -1):
            for length in range(1, (n - i) // 2 + 1):
                if lcp[i][i + length] >= length:
                    dp[i] = max(dp[i], 1 + dp[i + length])
        return dp[0]


def _test():
    s = Solution()
    assert s.deleteString("abcabcdabc") == 2
    assert s.deleteString("aaabaab") == 4
    assert s.deleteString("aaaaa") == 5
    print("LC 2430 OK")


if __name__ == "__main__":
    _test()

