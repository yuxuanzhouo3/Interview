"""
LC 3490. Count Beautiful Numbers — Hard
https://leetcode.com/problems/count-beautiful-numbers/
Tags: Dynamic Programming
Best: 最优解
"""
import functools

class Solution:
  def beautifulNumbers(self, l: int, r: int) -> int:
    @functools.lru_cache(None)
    def dp(
        s: str,
        i: int,
        tight: bool,
        isLeadingZero: bool,
        hasZero: bool,
        sum: int,
        prod: int,
    ) -> int:
      if i == len(s):
        if isLeadingZero:
          return 0
        return 1 if hasZero or prod % sum == 0 else 0
      if not isLeadingZero and hasZero and not tight:
        return 10 ** (len(s) - i)

      res = 0
      maxDigit = int(s[i]) if tight else 9

      for d in range(maxDigit + 1):
        nextTight = tight and (d == maxDigit)
        nextIsLeadingZero = isLeadingZero and d == 0
        nextHasZero = not nextIsLeadingZero and d == 0
        nextProd = 1 if nextIsLeadingZero else prod * d
        res += dp(s, i + 1, nextTight, nextIsLeadingZero,
                  nextHasZero, sum + d, nextProd)

      return res

    return (dp(str(r), 0, tight=True, isLeadingZero=True, hasZero=False, sum=0, prod=1) -
            dp(str(l - 1), 0, tight=True, isLeadingZero=True, hasZero=False, sum=0, prod=1))

def _test():
    s = Solution()
    assert s.beautifulNumbers(l=10, r=20) == 2
    assert s.beautifulNumbers(l=1, r=15) == 10
    print("LC 3490 OK")

if __name__ == "__main__":
    _test()
