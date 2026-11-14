"""
LC 1425. Constrained Subsequence Sum — Hard
https://leetcode.com/problems/constrained-subsequence-sum/
Best: DP + mono deque — O(n)
"""

from typing import List
from collections import deque


class Solution:
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = nums[:]
        dq = deque()  # indices of decreasing dp
        ans = nums[0]
        for i in range(n):
            while dq and dq[0] < i - k:
                dq.popleft()
            if dq:
                dp[i] = max(dp[i], nums[i] + dp[dq[0]])
            ans = max(ans, dp[i])
            while dq and dp[dq[-1]] <= dp[i]:
                dq.pop()
            dq.append(i)
        return ans


def _test():
    s = Solution()
    assert s.constrainedSubsetSum([10, 2, -10, 5, 20], 2) == 37
    assert s.constrainedSubsetSum([-1, -2, -3], 1) == -1
    assert s.constrainedSubsetSum([10, -2, -10, -5, 20], 2) == 23
    print("LC 1425 OK")


if __name__ == "__main__":
    _test()
