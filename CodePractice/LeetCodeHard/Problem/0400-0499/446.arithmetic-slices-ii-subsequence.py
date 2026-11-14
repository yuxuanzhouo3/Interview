"""
LC 446. Arithmetic Slices II - Subsequence — Hard
Best: DP map diff→count weak subsequences — O(n^2)
"""

from typing import List
from collections import defaultdict


class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [defaultdict(int) for _ in range(n)]
        ans = 0
        for i in range(n):
            for j in range(i):
                d = nums[i] - nums[j]
                cnt = dp[j][d]
                dp[i][d] += cnt + 1
                ans += cnt
        return ans


def _test():
    s = Solution()
    assert s.numberOfArithmeticSlices([2, 4, 6, 8, 10]) == 7
    assert s.numberOfArithmeticSlices([7, 7, 7, 7, 7]) == 16
    print("LC 446 OK")


if __name__ == "__main__":
    _test()
