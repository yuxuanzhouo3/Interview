"""
LC 2589. Minimum Time to Complete All Tasks — Hard
https://leetcode.com/problems/minimum-time-to-complete-all-tasks/
Best: Sort by end, greedy fill from right — O(nT) T≤2000
"""

from typing import List


class Solution:
    def findMinimumTime(self, tasks: List[List[int]]) -> int:
        tasks.sort(key=lambda x: x[1])
        on = [False] * 2001
        for start, end, duration in tasks:
            have = sum(1 for t in range(start, end + 1) if on[t])
            need = duration - have
            t = end
            while need > 0:
                if not on[t]:
                    on[t] = True
                    need -= 1
                t -= 1
        return sum(on)


def _test():
    s = Solution()
    assert s.findMinimumTime([[2, 3, 1], [4, 5, 1], [1, 5, 2]]) == 2
    assert s.findMinimumTime([[1, 3, 2], [2, 5, 3], [5, 6, 2]]) == 4
    print("LC 2589 OK")


if __name__ == "__main__":
    _test()
