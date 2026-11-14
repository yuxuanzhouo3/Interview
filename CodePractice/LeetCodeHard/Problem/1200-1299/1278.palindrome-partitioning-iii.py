"""
LC 1278. Palindrome Partitioning III — Hard
https://leetcode.com/problems/palindrome-partitioning-iii/
Best: Cost precompute + DP — O(n^2 k)
"""


class Solution:
    def palindromePartition(self, s: str, k: int) -> int:
        n = len(s)
        cost = [[0] * n for _ in range(n)]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                cost[i][j] = cost[i + 1][j - 1] + (s[i] != s[j])
        INF = 10**9
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            for p in range(1, min(k, i) + 1):
                for j in range(p - 1, i):
                    dp[i][p] = min(dp[i][p], dp[j][p - 1] + cost[j][i - 1])
        return dp[n][k]


def _test():
    s = Solution()
    assert s.palindromePartition("abc", 2) == 1
    assert s.palindromePartition("aabbc", 3) == 0
    assert s.palindromePartition("leetcode", 8) == 0
    print("LC 1278 OK")


if __name__ == "__main__":
    _test()
