"""
LC 3041. Maximize Consecutive Elements in an Array After Modification — Hard
https://leetcode.com/problems/maximize-consecutive-elements-in-an-array-after-modification/
Best: Sort + DP on value — O(n log n)
"""

from typing import List


class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        ans = 0
        for x in nums:
            a = dp.get(x - 1, 0) + 1
            b = dp.get(x, 0) + 1
            dp[x] = max(dp.get(x, 0), a)
            dp[x + 1] = max(dp.get(x + 1, 0), b)
            ans = max(ans, dp[x], dp[x + 1])
        return ans


def _test():
    s = Solution()
    assert s.maxSelectedElements([2, 1, 5, 1, 1]) == 3
    assert s.maxSelectedElements([1, 4, 7, 10]) == 1
    assert s.maxSelectedElements([8, 10, 6, 12, 9]) == 4
    print("LC 3041 OK")


if __name__ == "__main__":
    _test()
