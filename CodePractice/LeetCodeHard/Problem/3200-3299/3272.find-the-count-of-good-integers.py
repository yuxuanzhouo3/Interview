"""
LC 3272. Find the Count of Good Integers — Hard
https://leetcode.com/problems/find-the-count-of-good-integers/
Best: 方法一：枚举 + 组合数学
"""

from collections import Counter
from math import factorial

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        fac = [factorial(i) for i in range(n + 1)]
        ans = 0
        vis = set()
        base = 10 ** ((n - 1) // 2)
        for i in range(base, base * 10):
            s = str(i)
            s += s[::-1][n % 2 :]
            if int(s) % k:
                continue
            t = "".join(sorted(s))
            if t in vis:
                continue
            vis.add(t)
            cnt = Counter(t)
            res = (n - cnt["0"]) * fac[n - 1]
            for x in cnt.values():
                res //= fac[x]
            ans += res
        return ans

def _test():
    s = Solution()
    assert s.countGoodIntegers(n=3, k=5) == 27
    assert s.countGoodIntegers(n=1, k=4) == 2
    assert s.countGoodIntegers(n=5, k=6) == 2468
    print("LC 3272 OK")

if __name__ == "__main__":
    _test()
