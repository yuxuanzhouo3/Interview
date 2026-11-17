"""
LC 3470. Permutations IV — Hard
https://leetcode.com/problems/permutations-iv/
Tags: Array, Math, Combinatorics, Enumeration
Best: 最优解
"""
import math

class Solution:
  def permute(self, n: int, k: int) -> list[int]:
    ans = []
    isLookingForEven = True
    remainingNumbers = list(range(1, n + 1))

    for turn in range(n):
      remainingPermutations = (math.factorial((n - 1 - turn) // 2) *
                               math.factorial((n - turn) // 2))
      found = False
      for index, number in enumerate(remainingNumbers):
        if number % 2 != isLookingForEven and (turn > 0 or n % 2 == 1):
          continue
        if k <= remainingPermutations:
          ans.append(remainingNumbers.pop(index))
          isLookingForEven = ans[-1] % 2 == 0
          found = True
          break
        k -= remainingPermutations
      if not found:
        return []

    return ans

def _test():
    s = Solution()
    assert s.permute(n=4, k=6) == [3,4,1,2]
    assert s.permute(n=3, k=2) == [3,2,1]
    assert s.permute(n=2, k=3) == []
    print("LC 3470 OK")

if __name__ == "__main__":
    _test()
