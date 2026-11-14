"""
LC 699. Falling Squares — Hard
https://leetcode.com/problems/falling-squares/
Best: coordinate compress + segment tree / sorted intervals — O(n^2)
"""

from typing import List


class Solution:
    def fallingSquares(self, positions: List[List[int]]) -> List[int]:
        # intervals: (L, R, height)  R exclusive
        intervals = []  # list of [L, R, h]
        ans = []
        max_h = 0
        for left, side in positions:
            right = left + side
            base = 0
            for L, R, h in intervals:
                if R > left and L < right:
                    base = max(base, h)
            new_h = base + side
            intervals.append([left, right, new_h])
            max_h = max(max_h, new_h)
            ans.append(max_h)
        return ans


def _test():
    s = Solution()
    assert s.fallingSquares([[1, 2], [2, 3], [6, 1]]) == [2, 5, 5]
    assert s.fallingSquares([[100, 100], [200, 100]]) == [100, 100]
    print("LC 699 OK")


if __name__ == "__main__":
    _test()
