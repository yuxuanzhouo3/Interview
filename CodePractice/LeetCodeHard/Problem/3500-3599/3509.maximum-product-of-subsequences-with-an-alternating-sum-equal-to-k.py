"""
LC 3509. Maximum Product of Subsequences With an Alternating Sum Equal to K — Hard
https://leetcode.com/problems/maximum-product-of-subsequences-with-an-alternating-sum-equal-to-k/
Tags: Array, Hash Table, Dynamic Programming
Best: 最优解
"""
import functools

from enum import Enum


class State(Enum):
  FIRST = 0  # first element - add to sum and start product
  SUBTRACT = 1  # second element - subtract from sum and multiply product
  ADD = 2  # third element - add to sum and multiply product


class Solution:
  def maxProduct(self, nums: list[int], k: int, limit: int) -> int:
    MIN = -5000
    if abs(k) > sum(nums):
      return -1

    @functools.lru_cache(None)
    def dp(i: int, product: int, state: State, k: int) -> int:
      if i == len(nums):
        return (product if k == 0 and state != State.FIRST and product <= limit
                else MIN)
      res = dp(i + 1, product, state, k)
      if state == State.FIRST:
        res = max(res, dp(i + 1, nums[i], State.SUBTRACT, k - nums[i]))
      if state == State.SUBTRACT:
        res = max(res, dp(i + 1, min(product * nums[i], limit + 1),
                          State.ADD, k + nums[i]))
      if state == State.ADD:
        res = max(res, dp(i + 1, min(product * nums[i], limit + 1),
                          State.SUBTRACT, k - nums[i]))
      return res

    ans = dp(0, 1, State.FIRST, k)
    return -1 if ans == MIN else ans

def _test():
    s = Solution()
    assert s.maxProduct(nums=[1,2,3], k=2, limit=10) == 6
    assert s.maxProduct(nums=[0,2,3], k=-5, limit=12) == -1
    assert s.maxProduct(nums=[2,2,3,3], k=0, limit=9) == 9
    print("LC 3509 OK")

if __name__ == "__main__":
    _test()
