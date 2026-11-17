"""
LC 3500. Minimum Cost to Divide Array Into Subarrays — Hard
https://leetcode.com/problems/minimum-cost-to-divide-array-into-subarrays/
Tags: Array, Dynamic Programming, Prefix Sum, Li Chao Tree
Best: 最优解
"""
import math
import itertools
from itertools import accumulate

class Solution:
  def minimumCost(self, nums: list[int], cost: list[int], k: int) -> int:
    n = len(nums)
    prefixNums = list(itertools.accumulate(nums, initial=0))
    prefixCost = list(itertools.accumulate(cost, initial=0))
    # dp[i] := the minimum cost to divide nums[i..n - 1] into subarrays
    dp = [math.inf] * n + [0]

    for i in range(n - 1, -1, -1):
      for j in range(i, n):
        dp[i] = min(dp[i],
                    prefixNums[j + 1] * (prefixCost[j + 1] - prefixCost[i]) +
                    k * (prefixCost[n] - prefixCost[i]) + dp[j + 1])

    return dp[0]

def _test():
    s = Solution()
    assert s.minimumCost(nums=[3,1,4], cost=[4,6,6], k=1) == 110
    assert s.minimumCost(nums=[4,8,5,1,14,2,2,12,1], cost=[7,2,8,4,2,2,1,1,2], k=7) == 985
    print("LC 3500 OK")

if __name__ == "__main__":
    _test()
