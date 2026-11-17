"""
LC 2136. Earliest Possible Day of Full Bloom — Hard
https://leetcode.com/problems/earliest-possible-day-of-full-bloom/
Best: plant by growTime desc — O(n log n)
"""

from typing import List


class Solution:
    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        order = sorted(range(len(growTime)), key=lambda i: -growTime[i])
        day = 0
        ans = 0
        for i in order:
            day += plantTime[i]
            ans = max(ans, day + growTime[i])
        return ans


def _test():
    s = Solution()
    assert s.earliestFullBloom([1, 4, 3], [2, 3, 1]) == 9
    assert s.earliestFullBloom([1, 2, 3, 2], [2, 1, 2, 1]) == 9
    assert s.earliestFullBloom([1], [1]) == 2
    print("LC 2136 OK")


if __name__ == "__main__":
    _test()
