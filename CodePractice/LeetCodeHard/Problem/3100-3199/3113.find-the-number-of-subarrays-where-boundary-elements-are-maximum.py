"""
LC 3113. Find the Number of Subarrays Where Boundary Elements Are Maximum — Hard
https://leetcode.com/problems/find-the-number-of-subarrays-where-boundary-elements-are-maximum/
Best: Monotonic stack — O(n)
"""

from typing import List


class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        stk = []  # [value, count]
        ans = 0
        for x in nums:
            while stk and stk[-1][0] < x:
                stk.pop()
            if not stk or stk[-1][0] > x:
                stk.append([x, 1])
            else:
                stk[-1][1] += 1
            ans += stk[-1][1]
        return ans


def _test():
    s = Solution()
    assert s.numberOfSubarrays([1, 4, 3, 3, 2]) == 6
    assert s.numberOfSubarrays([3, 3, 3]) == 6
    assert s.numberOfSubarrays([1]) == 1
    print("LC 3113 OK")


if __name__ == "__main__":
    _test()
