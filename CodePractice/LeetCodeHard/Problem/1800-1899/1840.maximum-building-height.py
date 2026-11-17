"""
LC 1840. Maximum Building Height — Hard
https://leetcode.com/problems/maximum-building-height/
Best: L->R then R->L clamp + peak between — O(r log r)
"""

from typing import List


class Solution:
    def maxBuilding(self, n: int, restrictions: List[List[int]]) -> int:
        rs = [[1, 0]] + restrictions + [[n, n - 1]]
        rs.sort()
        m = len(rs)
        # left to right
        for i in range(1, m):
            gap = rs[i][0] - rs[i - 1][0]
            rs[i][1] = min(rs[i][1], rs[i - 1][1] + gap)
        # right to left
        for i in range(m - 2, -1, -1):
            gap = rs[i + 1][0] - rs[i][0]
            rs[i][1] = min(rs[i][1], rs[i + 1][1] + gap)
        ans = 0
        for i in range(1, m):
            lpos, lh = rs[i - 1]
            rpos, rh = rs[i]
            # max peak height between two buildings
            ans = max(ans, (rh + lh + rpos - lpos) // 2)
        return ans


def _test():
    s = Solution()
    assert s.maxBuilding(5, [[2, 1], [4, 1]]) == 2
    assert s.maxBuilding(6, []) == 5
    assert s.maxBuilding(10, [[5, 3], [2, 5], [7, 4], [10, 3]]) == 5
    print("LC 1840 OK")


if __name__ == "__main__":
    _test()
