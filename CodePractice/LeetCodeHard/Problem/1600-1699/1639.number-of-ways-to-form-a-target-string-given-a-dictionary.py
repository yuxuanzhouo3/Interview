"""
LC 1639. Number of Ways to Form a Target String Given a Dictionary — Hard
https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/
Best: count[col][char] + DP — O(m*(n+|Σ|))
"""

from typing import List


class Solution:
    def numWays(self, words: List[str], target: str) -> int:
        MOD = 10**9 + 7
        m = len(words[0])
        tlen = len(target)
        count = [[0] * 26 for _ in range(m)]
        for w in words:
            for j, ch in enumerate(w):
                count[j][ord(ch) - 97] += 1

        # dp[i][j]: ways to form target[:i] using first j columns
        dp = [[0] * (m + 1) for _ in range(tlen + 1)]
        for j in range(m + 1):
            dp[0][j] = 1
        for i in range(1, tlen + 1):
            c = ord(target[i - 1]) - 97
            for j in range(1, m + 1):
                dp[i][j] = dp[i][j - 1]
                if count[j - 1][c]:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1] * count[j - 1][c]) % MOD
        return dp[tlen][m]


def _test():
    s = Solution()
    assert s.numWays(["acca", "bbbb", "caca"], "aba") == 6
    assert s.numWays(["abba", "baab"], "bab") == 4
    print("LC 1639 OK")


if __name__ == "__main__":
    _test()
