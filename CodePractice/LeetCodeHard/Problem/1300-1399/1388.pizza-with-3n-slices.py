"""
LC 1388. Pizza With 3n Slices — Hard
https://leetcode.com/problems/pizza-with-3n-slices/
Best: House robber on circle, take n slices — O(n^2)
"""

from typing import List


class Solution:
    def maxSizeSlices(self, slices: List[int]) -> int:
        def rob(arr: List[int], k: int) -> int:
            m = len(arr)
            # dp[i][j] = max using first i elements, take j
            dp = [[0] * (k + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, k + 1):
                    dp[i][j] = max(dp[i - 1][j], (dp[i - 2][j - 1] if i >= 2 else 0) + arr[i - 1])
            return dp[m][k]

        n = len(slices) // 3
        return max(rob(slices[:-1], n), rob(slices[1:], n))


def _test():
    s = Solution()
    assert s.maxSizeSlices([1, 2, 3, 4, 5, 6]) == 10
    assert s.maxSizeSlices([8, 9, 8, 6, 1, 1]) == 16
    assert s.maxSizeSlices([4, 1, 2, 5, 8, 3, 1, 9, 7]) == 21
    print("LC 1388 OK")


if __name__ == "__main__":
    _test()
