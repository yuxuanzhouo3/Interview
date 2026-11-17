"""
LC 1755. Closest Subsequence Sum — Hard
https://leetcode.com/problems/closest-subsequence-sum/
Best: Meet in the middle — O(2^{n/2} n)
"""

from typing import List
import bisect


class Solution:
    def minAbsDifference(self, nums: List[int], goal: int) -> int:
        n = len(nums)
        mid = n // 2

        def all_sums(arr):
            res = {0}
            for x in arr:
                res |= {s + x for s in res}
            return sorted(res)

        left = all_sums(nums[:mid])
        right = all_sums(nums[mid:])
        ans = abs(goal)
        for s in left:
            t = goal - s
            i = bisect.bisect_left(right, t)
            if i < len(right):
                ans = min(ans, abs(s + right[i] - goal))
            if i:
                ans = min(ans, abs(s + right[i - 1] - goal))
        return ans


def _test():
    s = Solution()
    assert s.minAbsDifference([5, -7, 3, 5], 6) == 0
    assert s.minAbsDifference([7, -9, 15, -2], -5) == 1
    assert s.minAbsDifference([1, 2, 3], -7) == 7
    print("LC 1755 OK")


if __name__ == "__main__":
    _test()
