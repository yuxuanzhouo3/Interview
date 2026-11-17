"""
LC 1691. Maximum Height by Stacking Cuboids — Hard
https://leetcode.com/problems/maximum-height-by-stacking-cuboids/
Best: sort dims + LIS-like DP on height — O(n^2)
"""

from typing import List


class Solution:
    def maxHeight(self, cuboids: List[List[int]]) -> int:
        for c in cuboids:
            c.sort()
        cuboids.sort()
        n = len(cuboids)
        dp = [0] * n
        for i in range(n):
            dp[i] = cuboids[i][2]
            for j in range(i):
                if (
                    cuboids[j][0] <= cuboids[i][0]
                    and cuboids[j][1] <= cuboids[i][1]
                    and cuboids[j][2] <= cuboids[i][2]
                ):
                    dp[i] = max(dp[i], dp[j] + cuboids[i][2])
        return max(dp)


def _test():
    s = Solution()
    assert s.maxHeight([[50, 45, 20], [95, 37, 53], [45, 23, 12]]) == 190
    assert s.maxHeight([[38, 25, 45], [76, 35, 27]]) == 76
    assert (
        s.maxHeight(
            [
                [7, 11, 17],
                [7, 17, 11],
                [11, 7, 17],
                [11, 17, 7],
                [17, 7, 11],
                [17, 11, 7],
            ]
        )
        == 102
    )
    print("LC 1691 OK")


if __name__ == "__main__":
    _test()
