"""
LC 1977. Number of Ways to Separate Numbers — Hard
https://leetcode.com/problems/number-of-ways-to-separate-numbers/
Best: LCP + 前缀和 DP — O(n^2)
"""


class Solution:
    def numberOfCombinations(self, num: str) -> int:
        MOD = 10**9 + 7
        n = len(num)
        if num[0] == "0":
            return 0
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if num[i] == num[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1
        # dp[i][j] = ways for num[:i] last length j; also keep prefix in same array
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                v = 0
                if num[i - j] != "0":
                    if i - 2 * j >= 0:
                        a, b = i - 2 * j, i - j
                        c = lcp[a][b]
                        if c >= j or num[a + c] <= num[b + c]:
                            v = dp[i - j][j]
                        else:
                            v = dp[i - j][min(j - 1, i - j)]
                    else:
                        v = dp[i - j][min(j - 1, i - j)] if i - j > 0 else dp[0][0]
                        if i - j == 0:
                            v = dp[0][0]
                dp[i][j] = (dp[i][j - 1] + v) % MOD
        return dp[n][n]


def _test():
    s = Solution()
    assert s.numberOfCombinations("327") == 2
    assert s.numberOfCombinations("094") == 0
    assert s.numberOfCombinations("0") == 0
    assert s.numberOfCombinations("9999999999999") == 101
    print("LC 1977 OK")


if __name__ == "__main__":
    _test()
