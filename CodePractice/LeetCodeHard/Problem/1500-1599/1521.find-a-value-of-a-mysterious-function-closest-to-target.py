"""
LC 1521. Find a Value of a Mysterious Function Closest to Target — Hard
https://leetcode.com/problems/find-a-value-of-a-mysterious-function-closest-to-target/
Best: Maintain set of AND values ending at i — O(n · 30)
"""

from typing import List


class Solution:
    def closestToTarget(self, arr: List[int], target: int) -> int:
        ans = abs(arr[0] - target)
        cur = set()
        for x in arr:
            cur = {x & y for y in cur} | {x}
            for v in cur:
                ans = min(ans, abs(v - target))
        return ans


def _test():
    s = Solution()
    assert s.closestToTarget([9, 12, 3, 7, 15], 5) == 2
    assert s.closestToTarget([1000000, 1000000, 1000000], 1) == 999999
    assert s.closestToTarget([1, 2, 4, 8, 16], 0) == 0
    print("LC 1521 OK")


if __name__ == "__main__":
    _test()
