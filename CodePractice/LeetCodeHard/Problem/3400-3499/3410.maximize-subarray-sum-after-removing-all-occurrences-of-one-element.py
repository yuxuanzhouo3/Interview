"""
LC 3410. Maximize Subarray Sum After Removing All Occurrences of One Element — Hard
https://leetcode.com/problems/maximize-subarray-sum-after-removing-all-occurrences-of-one-element/
Tags: Array, Hash Table, Divide and Conquer, Dynamic Programming, Segment Tree, Prefix Sum
Best: 最优解
"""
import collections

class Solution:
  def maxSubarraySum(self, nums: list[int]) -> int:
    ans = max(nums)
    prefix = 0
    minPrefix = 0
    # the minimum prefix sum that can have a negative number removed
    modifiedMinPrefix = 0
    count = collections.Counter()
    # minPrefixPlusRemoval[num] := the minimum prefix sum plus removed `num`
    minPrefixPlusRemoval = {}

    for num in nums:
      prefix += num
      ans = max(ans, prefix - modifiedMinPrefix)
      if num < 0:
        count[num] += 1
        minPrefixPlusRemoval[num] = (
            min(minPrefixPlusRemoval.get(num, 0), minPrefix) + num)
        modifiedMinPrefix = min(modifiedMinPrefix,
                                count[num] * num,
                                minPrefixPlusRemoval[num])
      minPrefix = min(minPrefix, prefix)
      modifiedMinPrefix = min(modifiedMinPrefix, minPrefix)

    return ans

def _test():
    s = Solution()
    assert s.maxSubarraySum(nums=[-3,2,-2,-1,3,-2,3]) == 7
    assert s.maxSubarraySum(nums=[1,2,3,4]) == 10
    print("LC 3410 OK")

if __name__ == "__main__":
    _test()
