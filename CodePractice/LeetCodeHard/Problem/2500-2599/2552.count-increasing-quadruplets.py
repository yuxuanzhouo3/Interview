"""
LC 2552. Count Increasing Quadruplets — Hard
https://leetcode.com/problems/count-increasing-quadruplets/
Best: O(n^2) dp counting — O(n^2)
"""

from typing import List


class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        dp = [0] * n  # dp[j] = # of (i,k) with i<j<k, nums[i]<nums[k]<nums[j] for k already seen as current mid...
        for k in range(n):
            less = 0
            for j in range(k):
                if nums[j] < nums[k]:
                    less += 1
                    ans += dp[j]
                elif nums[j] > nums[k]:
                    dp[j] += less
        return ans


def _test():
    s = Solution()
    assert s.countQuadruplets([1, 3, 2, 4, 5]) == 2
    assert s.countQuadruplets([1, 2, 3, 4]) == 0
    print("LC 2552 OK")


if __name__ == "__main__":
    _test()

