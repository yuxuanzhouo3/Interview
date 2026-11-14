"""
LC 568. Maximum Vacation Days — Hard
https://leetcode.com/problems/maximum-vacation-days/
Best: DP cities × weeks — O(K * N^2)
"""

from typing import List


class Solution:
    def maxVacationDays(self, flights: List[List[int]], days: List[List[int]]) -> int:
        n = len(days)
        k = len(days[0]) if n else 0
        # can[i][j] = flights or stay
        reach = [[False] * n for _ in range(n)]
        for i in range(n):
            reach[i][i] = True
            for j in range(n):
                if flights[i][j]:
                    reach[i][j] = True

        NEG = -10**9
        # dp[city] after processing weeks; start week0 from city0 only (or fly from 0)
        dp = [NEG] * n
        dp[0] = 0
        for week in range(k):
            ndp = [NEG] * n
            for c in range(n):
                if dp[c] == NEG:
                    continue
                for d in range(n):
                    if reach[c][d]:
                        ndp[d] = max(ndp[d], dp[c] + days[d][week])
            dp = ndp
        return max(dp) if max(dp) > NEG else 0


def _test():
    s = Solution()
    assert s.maxVacationDays([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[1, 3, 1], [6, 0, 3], [3, 3, 3]]) == 12
    assert s.maxVacationDays([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [7, 7, 7], [7, 7, 7]]) == 3
    assert s.maxVacationDays([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[7, 0, 0], [0, 7, 0], [0, 0, 7]]) == 21
    print("LC 568 OK")


if __name__ == "__main__":
    _test()
