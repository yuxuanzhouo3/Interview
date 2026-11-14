"""
LC 2454. Next Greater Element IV — Hard
https://leetcode.com/problems/next-greater-element-iv/
Best: Monotonic stack of (value, idx) with second-greater — O(n)
"""

from typing import List


class Solution:
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [-1] * n
        s1, s2 = [], []  # decreasing stacks of indices
        for i, x in enumerate(nums):
            while s2 and nums[s2[-1]] < x:
                ans[s2.pop()] = x
            tmp = []
            while s1 and nums[s1[-1]] < x:
                tmp.append(s1.pop())
            s2.extend(reversed(tmp))
            s1.append(i)
        return ans


def _test():
    s = Solution()
    assert s.secondGreaterElement([2, 4, 0, 9, 6]) == [9, 6, 6, -1, -1]
    assert s.secondGreaterElement([3, 3]) == [-1, -1]
    print("LC 2454 OK")


if __name__ == "__main__":
    _test()
