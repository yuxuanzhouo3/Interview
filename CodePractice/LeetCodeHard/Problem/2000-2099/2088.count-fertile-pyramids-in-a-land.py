"""
LC 2088. Count Fertile Pyramids in a Land — Hard
https://leetcode.com/problems/count-fertile-pyramids-in-a-land/
Best: DP pyramid height both orientations — O(mn)
"""

from typing import List


class Solution:
    def countPyramids(self, grid: List[List[int]]) -> int:
        def count(g: List[List[int]]) -> int:
            m, n = len(g), len(g[0])
            dp = [row[:] for row in g]
            ans = 0
            for i in range(m - 2, -1, -1):
                for j in range(1, n - 1):
                    if g[i][j] == 1:
                        dp[i][j] = min(dp[i + 1][j - 1], dp[i + 1][j], dp[i + 1][j + 1]) + 1
                        ans += dp[i][j] - 1
                    else:
                        dp[i][j] = 0
            return ans

        return count(grid) + count(grid[::-1])


def _test():
    s = Solution()
    assert s.countPyramids([[0, 1, 1, 0], [1, 1, 1, 1]]) == 2
    assert s.countPyramids([[1, 1, 1], [1, 1, 1]]) == 2
    assert s.countPyramids([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 0, 0, 1]]) == 13
    print("LC 2088 OK")


if __name__ == "__main__":
    _test()
