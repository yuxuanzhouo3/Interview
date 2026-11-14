"""
LC 664. Strange Printer — Hard
https://leetcode.com/problems/strange-printer/
Best: interval DP — O(n^3)
"""


class Solution:
    def strangePrinter(self, s: str) -> int:
        # compress consecutive duplicates
        t = []
        for c in s:
            if not t or t[-1] != c:
                t.append(c)
        n = len(t)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = dp[i + 1][j] + 1
                for k in range(i + 1, j + 1):
                    if t[k] == t[i]:
                        dp[i][j] = min(
                            dp[i][j],
                            dp[i][k - 1] + (dp[k + 1][j] if k + 1 <= j else 0),
                        )
                # also standard split
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j])
        return dp[0][n - 1]


def _test():
    s = Solution()
    assert s.strangePrinter("aaabbb") == 2
    assert s.strangePrinter("aba") == 2
    assert s.strangePrinter("abc") == 3
    print("LC 664 OK")


if __name__ == "__main__":
    _test()
