"""
LC 42. Trapping Rain Water — Hard
Best: Two pointers — O(n) time / O(1) space
"""

from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n < 3:
            return 0
        l, r = 0, n - 1
        left_max = right_max = 0
        ans = 0
        while l < r:
            if height[l] <= height[r]:
                if height[l] >= left_max:
                    left_max = height[l]
                else:
                    ans += left_max - height[l]
                l += 1
            else:
                if height[r] >= right_max:
                    right_max = height[r]
                else:
                    ans += right_max - height[r]
                r -= 1
        return ans


def _test():
    s = Solution()
    assert s.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert s.trap([4, 2, 0, 3, 2, 5]) == 9
    print("LC 42 OK")


if __name__ == "__main__":
    _test()
