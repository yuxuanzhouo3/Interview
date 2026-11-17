"""
LC 2463. Minimum Total Distance Traveled — Hard
https://leetcode.com/problems/minimum-total-distance-traveled/
Best: Sort + DP assign robots to factories — O(R*F*cap)
"""


from typing import List


class Solution:
    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot.sort()
        factory.sort()
        # expand factories into repair slots
        slots = []
        for pos, lim in factory:
            slots.extend([pos] * lim)
        r, f = len(robot), len(slots)
        INF = 10**30
        dp = [[INF] * (f + 1) for _ in range(r + 1)]
        for j in range(f + 1):
            dp[0][j] = 0
        for i in range(1, r + 1):
            for j in range(1, f + 1):
                dp[i][j] = dp[i][j - 1]  # skip slot
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + abs(robot[i - 1] - slots[j - 1]))
        return dp[r][f]


def _test():
    s = Solution()
    assert s.minimumTotalDistance([0, 4, 6], [[2, 2], [6, 2]]) == 4
    assert s.minimumTotalDistance([1, -1], [[-2, 1], [2, 1]]) == 2
    print("LC 2463 OK")


if __name__ == "__main__":
    _test()

