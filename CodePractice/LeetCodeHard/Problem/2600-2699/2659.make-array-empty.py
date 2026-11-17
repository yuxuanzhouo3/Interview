"""
LC 2659. Make Array Empty — Hard
https://leetcode.com/problems/make-array-empty/
Best: Sort by value + index rotate count — O(n log n)
"""

from typing import List


class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        n = len(nums)
        pos = {v: i for i, v in enumerate(nums)}
        vals = sorted(nums)
        ans = n
        for i in range(1, n):
            if pos[vals[i]] < pos[vals[i - 1]]:
                ans += n - i
        return ans


def _test():
    s = Solution()
    assert s.countOperationsToEmptyArray([3, 1, 2]) == 4
    assert s.countOperationsToEmptyArray([1, 2, 3, 4]) == 4
    assert s.countOperationsToEmptyArray([1, 2, 4, 3]) == 5
    print("LC 2659 OK")


if __name__ == "__main__":
    _test()
