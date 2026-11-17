"""
LC 1416. Restore The Array — Hard
https://leetcode.com/problems/restore-the-array/
Best: DP + digit DP style — O(n · log k)
"""


class Solution:
    def numberOfArrays(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        dp = [0] * (n + 1)
        dp[n] = 1
        for i in range(n - 1, -1, -1):
            if s[i] == "0":
                continue
            num = 0
            for j in range(i, n):
                num = num * 10 + ord(s[j]) - 48
                if num > k:
                    break
                dp[i] = (dp[i] + dp[j + 1]) % MOD
        return dp[0]


def _test():
    s = Solution()
    assert s.numberOfArrays("1000", 10000) == 1
    assert s.numberOfArrays("1000", 10) == 0
    assert s.numberOfArrays("1317", 2000) == 8
    assert s.numberOfArrays("2020", 30) == 1
    print("LC 1416 OK")


if __name__ == "__main__":
    _test()
