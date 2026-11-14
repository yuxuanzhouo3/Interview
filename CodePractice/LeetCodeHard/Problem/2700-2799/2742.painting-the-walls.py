"""
LC 2742. Painting the Walls — Hard
https://leetcode.com/problems/painting-the-walls/
Best: 0-1 knapsack DP on paid painter coverage — O(n^2)
"""

from typing import List


class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        INF = 10**18
        dp = [INF] * (n + 1)
        dp[0] = 0
        for c, t in zip(cost, time):
            cover = t + 1  # paid paints 1, free paints t
            for j in range(n, -1, -1):
                nj = min(n, j + cover)
                dp[nj] = min(dp[nj], dp[j] + c)
        return dp[n]


def _test():
    s = Solution()
    assert s.paintWalls([1, 2, 3, 2], [1, 1, 1, 1]) == 3
    assert s.paintWalls([1, 2, 3, 2], [1, 1, 1, 1]) == 3
    assert s.paintWalls([2, 3, 4, 2], [1, 1, 1, 1]) == 4
    print("LC 2742 OK")


if __name__ == "__main__":
    _test()
