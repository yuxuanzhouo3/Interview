"""
LC 1402. Reducing Dishes — Hard
https://leetcode.com/problems/reducing-dishes/
Best: Sort + greedy/suffix — O(n log n)
"""

from typing import List


class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        satisfaction.sort()
        ans = cur = s = 0
        for x in reversed(satisfaction):
            s += x
            cur += s
            ans = max(ans, cur)
        return ans


def _test():
    s = Solution()
    assert s.maxSatisfaction([-1, -8, 0, 5, -9]) == 14
    assert s.maxSatisfaction([4, 3, 2]) == 20
    assert s.maxSatisfaction([-1, -4, -5]) == 0
    print("LC 1402 OK")


if __name__ == "__main__":
    _test()
