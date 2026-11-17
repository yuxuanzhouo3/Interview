"""
LC 3533. Concatenated Divisibility — Hard
https://leetcode.com/problems/concatenated-divisibility/
Tags: Array, Dynamic Programming, Bit Manipulation, Bitmask
Best: 最优解
"""
import functools

class Solution:
  def concatenatedDivisibility(self, nums: list[int], k: int) -> list[int]:
    n = len(nums)
    nums.sort()
    lengths = [len(str(num)) for num in nums]
    pows = [pow(10, length, k) for length in lengths]

    @functools.lru_cache(None)
    def dp(mask: int, mod: int) -> bool:
      """
      Returns True if there is a way to form a number divisible by `k` using the
      numbers in `nums`, where nums[i] is used iff `mask & (1 << i)`.
      """
      if mask == (1 << n) - 1:
        return mod == 0
      for i in range(n):
        if (mask >> i & 1) == 0:
          newMod = (mod * pows[i] + nums[i]) % k
          if dp(mask | 1 << i, newMod):
            return True
      return False

    def reconstruct(mask: int, mod: int) -> list[int]:
      """
      Reconstructs the numbers that form a number divisible by `k` using the
      numbers in `nums`, where nums[i] is used iff `mask & (1 << i)`.
      """
      for i in range(n):
        if (mask >> i & 1) == 0:
          newMod = (mod * pows[i] + nums[i]) % k
          if dp(mask | 1 << i, newMod):
            return [nums[i]] + reconstruct(mask | 1 << i, newMod)
      return []

    return reconstruct(0, 0) if dp(0, 0) else []

def _test():
    s = Solution()
    assert s.concatenatedDivisibility(nums=[3,12,45], k=5) == [3,12,45]
    assert s.concatenatedDivisibility(nums=[10,5], k=10) == [5,10]
    assert s.concatenatedDivisibility(nums=[1,2,3], k=5) == []
    print("LC 3533 OK")

if __name__ == "__main__":
    _test()
