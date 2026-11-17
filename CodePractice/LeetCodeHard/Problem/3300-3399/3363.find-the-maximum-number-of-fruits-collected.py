"""
LC 3363. Find the Maximum Number of Fruits Collected — Hard
https://leetcode.com/problems/find-the-maximum-number-of-fruits-collected/
Best: Diagonal + two triangle DPs — O(n^2)
"""

from typing import List


class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        ans = sum(fruits[i][i] for i in range(n))

        def top_right() -> int:
            dp = [float("-inf")] * n
            dp[n - 1] = fruits[0][n - 1]
            for i in range(1, n - 1):
                ndp = [float("-inf")] * n
                for j in range(i + 1, n):
                    best = max(
                        dp[j - 1] if j - 1 >= 0 else float("-inf"),
                        dp[j],
                        dp[j + 1] if j + 1 < n else float("-inf"),
                    )
                    if best > float("-inf"):
                        ndp[j] = best + fruits[i][j]
                dp = ndp
            best = float("-inf")
            for j in range(n):
                if abs(j - (n - 1)) <= 1:
                    best = max(best, dp[j])
            return 0 if best == float("-inf") else int(best)

        def bottom_left() -> int:
            dp = [float("-inf")] * n
            dp[n - 1] = fruits[n - 1][0]
            for j in range(1, n - 1):
                ndp = [float("-inf")] * n
                for i in range(j + 1, n):
                    best = max(
                        dp[i - 1] if i - 1 >= 0 else float("-inf"),
                        dp[i],
                        dp[i + 1] if i + 1 < n else float("-inf"),
                    )
                    if best > float("-inf"):
                        ndp[i] = best + fruits[i][j]
                dp = ndp
            best = float("-inf")
            for i in range(n):
                if abs(i - (n - 1)) <= 1:
                    best = max(best, dp[i])
            return 0 if best == float("-inf") else int(best)

        return ans + top_right() + bottom_left()


def _test():
    s = Solution()
    assert (
        s.maxCollectedFruits(
            [[1, 2, 3, 4], [5, 6, 8, 7], [9, 10, 11, 12], [13, 14, 15, 16]]
        )
        == 100
    )
    assert s.maxCollectedFruits([[1, 1], [1, 1]]) == 4
    print("LC 3363 OK")


if __name__ == "__main__":
    _test()
