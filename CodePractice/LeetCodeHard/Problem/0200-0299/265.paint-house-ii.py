"""
LC 265. Paint House II — Hard
https://leetcode.com/problems/paint-house-ii/
Best: 维护上一行最小/次小代价 — O(nk)
"""

from typing import List


class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        if not costs:
            return 0
        n, k = len(costs), len(costs[0])
        # dp[j] = 刷到当前房用颜色 j 的最小代价
        dp = costs[0][:]
        for i in range(1, n):
            # 找上一行最小、次小及最小颜色下标
            m1 = m2 = float("inf")
            c1 = -1
            for j, v in enumerate(dp):
                if v < m1:
                    m2, m1, c1 = m1, v, j
                elif v < m2:
                    m2 = v
            ndp = [0] * k
            for j in range(k):
                ndp[j] = costs[i][j] + (m2 if j == c1 else m1)
            dp = ndp
        return min(dp)


def _test():
    s = Solution()
    assert s.minCostII([[1, 5, 3], [2, 9, 4]]) == 5  # 1+4
    assert s.minCostII([[1, 3], [2, 4]]) == 5
    assert s.minCostII([[7, 6, 2]]) == 2
    print("LC 265 OK")


if __name__ == "__main__":
    _test()
