"""
LC 1246. Palindrome Removal — Hard
https://leetcode.com/problems/palindrome-removal/
Best: Interval DP — O(n^3)
"""

from typing import List


class Solution:
    def minimumMoves(self, arr: List[int]) -> int:
        n = len(arr)
        dp = [[n] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for i in range(n - 1):
            dp[i][i + 1] = 1 if arr[i] == arr[i + 1] else 2
        for d in range(2, n):
            for i in range(n - d):
                j = i + d
                if arr[i] == arr[j]:
                    dp[i][j] = dp[i + 1][j - 1]
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j])
        return dp[0][n - 1]


def _test():
    s = Solution()
    assert s.minimumMoves([1, 2]) == 2
    assert s.minimumMoves([1, 3, 4, 1, 5]) == 3
    assert s.minimumMoves([1, 1]) == 1
    assert s.minimumMoves([1]) == 1
    print("LC 1246 OK")


if __name__ == "__main__":
    _test()
