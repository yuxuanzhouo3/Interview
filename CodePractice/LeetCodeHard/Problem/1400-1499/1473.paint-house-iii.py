"""
LC 1473. Paint House III — Hard
https://leetcode.com/problems/paint-house-iii/
Best: DP houses × neighborhoods × color — O(n·m^2·target)
"""

from typing import List


class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        INF = 10**18
        # dp[j][c] = min cost with j neighborhoods, last color c (1..n)
        dp = [[INF] * (n + 1) for _ in range(target + 1)]
        dp[0][0] = 0  # sentinel: 0 neighborhoods, color 0
        for i in range(m):
            ndp = [[INF] * (n + 1) for _ in range(target + 1)]
            colors = range(1, n + 1) if houses[i] == 0 else [houses[i]]
            for j in range(target + 1):
                for pc in range(n + 1):
                    if dp[j][pc] >= INF:
                        continue
                    for c in colors:
                        nj = j + (0 if c == pc else 1)
                        if nj > target:
                            continue
                        add = 0 if houses[i] else cost[i][c - 1]
                        ndp[nj][c] = min(ndp[nj][c], dp[j][pc] + add)
            dp = ndp
        ans = min(dp[target][c] for c in range(1, n + 1))
        return -1 if ans >= INF else ans


def _test():
    s = Solution()
    assert s.minCost([0, 0, 0, 0, 0], [[1, 10], [10, 1], [10, 1], [1, 10], [5, 1]], 5, 2, 3) == 9
    assert s.minCost([0, 2, 1, 2, 0], [[1, 10], [10, 1], [10, 1], [1, 10], [5, 1]], 5, 2, 3) == 11
    assert s.minCost([3, 1, 2, 3], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], 4, 3, 3) == -1
    print("LC 1473 OK")


if __name__ == "__main__":
    _test()
