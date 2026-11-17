"""
LC 2398. Maximum Number of Robots Within Budget — Hard
https://leetcode.com/problems/maximum-number-of-robots-within-budget/
Best: Sliding window + mono deque — O(n)
"""

from typing import List
from collections import deque


class Solution:
    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
        n = len(chargeTimes)
        dq = deque()
        i = 0
        s = 0
        ans = 0
        for j in range(n):
            while dq and chargeTimes[dq[-1]] <= chargeTimes[j]:
                dq.pop()
            dq.append(j)
            s += runningCosts[j]
            while i <= j and chargeTimes[dq[0]] + (j - i + 1) * s > budget:
                if dq[0] == i:
                    dq.popleft()
                s -= runningCosts[i]
                i += 1
            ans = max(ans, j - i + 1)
        return ans


def _test():
    s = Solution()
    assert s.maximumRobots([3, 6, 1, 3, 4], [2, 1, 3, 4, 5], 25) == 3
    assert s.maximumRobots([11, 12, 19], [10, 8, 7], 19) == 0
    print("LC 2398 OK")


if __name__ == "__main__":
    _test()
