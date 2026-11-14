"""
LC 546. Remove Boxes — Hard
https://leetcode.com/problems/remove-boxes/
Best: Interval DP with trailing same-color count — O(n^4)
"""

from typing import List
from functools import lru_cache


class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        n = len(boxes)

        @lru_cache(None)
        def dp(l: int, r: int, k: int) -> int:
            """Max points removing boxes[l..r] with k extra boxes equal to boxes[r] after r."""
            if l > r:
                return 0
            # squeeze trailing same as boxes[r]
            while l < r and boxes[r - 1] == boxes[r]:
                r -= 1
                k += 1
            # remove the trailing group together
            ans = dp(l, r - 1, 0) + (k + 1) ** 2
            for i in range(l, r):
                if boxes[i] == boxes[r]:
                    ans = max(ans, dp(l, i, k + 1) + dp(i + 1, r - 1, 0))
            return ans

        return dp(0, n - 1, 0)


def _test():
    s = Solution()
    assert s.removeBoxes([1, 3, 2, 2, 2, 3, 4, 3, 1]) == 23
    assert s.removeBoxes([1, 1, 1]) == 9
    assert s.removeBoxes([1]) == 1
    print("LC 546 OK")


if __name__ == "__main__":
    _test()
