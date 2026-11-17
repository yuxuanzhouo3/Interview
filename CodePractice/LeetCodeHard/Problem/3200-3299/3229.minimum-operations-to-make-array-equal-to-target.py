"""
LC 3229. Minimum Operations to Make Array Equal to Target — Hard
https://leetcode.com/problems/minimum-operations-to-make-array-equal-to-target/
Best: Diff array greedy on positive increments — O(n)
"""

from typing import List


class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # need d[i] = target[i]-nums[i]; ops add/sub 1 on subarray
        # answer = sum of positive differences of consecutive d (with d[-1]=0)
        prev = 0
        ans = 0
        for a, b in zip(nums, target):
            cur = b - a
            if cur > prev:
                ans += cur - prev
            prev = cur
        return ans


def _test():
    s = Solution()
    assert s.minimumOperations([3, 5, 1, 2], [4, 6, 2, 4]) == 2
    assert s.minimumOperations([1, 3, 2], [2, 1, 4]) == 5
    print("LC 3229 OK")


if __name__ == "__main__":
    _test()
