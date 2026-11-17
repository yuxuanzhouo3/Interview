"""
LC 3357. Minimize the Maximum Adjacent Element Difference — Hard
https://leetcode.com/problems/minimize-the-maximum-adjacent-element-difference/
Best: Binary search answer + gap check with two values — O(n log M)
"""

from typing import List


class Solution:
    def minDifference(self, nums: List[int]) -> int:
        max_gap = 0
        mn, mx = 10**9, 0
        for i in range(1, len(nums)):
            a, b = nums[i - 1], nums[i]
            if (a == -1) != (b == -1):
                positive = max(a, b)
                mn = min(mn, positive)
                mx = max(mx, positive)
            else:
                max_gap = max(max_gap, abs(a - b))

        lo, hi = max_gap, (mx - mn + 1) // 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self._check(nums, mid, mn + mid, mx - mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _check(self, nums: List[int], m: int, x: int, y: int) -> bool:
        gap_len = 0
        prev = 0
        for num in nums:
            if num == -1:
                gap_len += 1
                continue
            if prev > 0 and gap_len > 0:
                if gap_len == 1 and not self._single(prev, num, m, x, y):
                    return False
                if gap_len > 1 and not self._multi(prev, num, m, x, y):
                    return False
            prev = num
            gap_len = 0
        if nums[0] == -1:
            num = next((v for v in nums if v != -1), -1)
            if num != -1 and not self._bound(num, m, x, y):
                return False
        if nums[-1] == -1:
            num = next((v for v in reversed(nums) if v != -1), -1)
            if num != -1 and not self._bound(num, m, x, y):
                return False
        return True

    def _single(self, a, b, m, x, y) -> bool:
        return min(max(abs(a - x), abs(b - x)), max(abs(a - y), abs(b - y))) <= m

    def _multi(self, a, b, m, x, y) -> bool:
        ax, ay, bx, by, xy = abs(a - x), abs(a - y), abs(b - x), abs(b - y), abs(x - y)
        return min(max(ax, bx), max(ay, by), max(ax, xy, by), max(ay, xy, bx)) <= m

    def _bound(self, a, m, x, y) -> bool:
        return min(abs(a - x), abs(a - y)) <= m


def _test():
    s = Solution()
    assert s.minDifference([1, 2, -1, 10, 8]) == 4
    assert s.minDifference([-1, -1, -1]) == 0
    assert s.minDifference([-1, 10, -1, 8]) == 1
    print("LC 3357 OK")


if __name__ == "__main__":
    _test()
