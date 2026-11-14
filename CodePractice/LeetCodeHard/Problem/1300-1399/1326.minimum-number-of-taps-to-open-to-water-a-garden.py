"""
LC 1326. Minimum Number of Taps to Open to Water a Garden — Hard
https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/
Best: Jump-game style greedy — O(n)
"""

from typing import List


class Solution:
    def minTaps(self, n: int, ranges: List[int]) -> int:
        reach = [0] * (n + 1)
        for i, r in enumerate(ranges):
            left = max(0, i - r)
            right = min(n, i + r)
            reach[left] = max(reach[left], right)
        ans = end = far = 0
        i = 0
        while end < n:
            while i <= end:
                far = max(far, reach[i])
                i += 1
            if far <= end:
                return -1
            ans += 1
            end = far
        return ans


def _test():
    s = Solution()
    assert s.minTaps(5, [3, 4, 1, 1, 0, 0]) == 1
    assert s.minTaps(3, [0, 0, 0, 0]) == -1
    print("LC 1326 OK")


if __name__ == "__main__":
    _test()
