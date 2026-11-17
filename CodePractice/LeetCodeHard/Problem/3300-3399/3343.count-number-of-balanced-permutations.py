"""
LC 3343. Count Number of Balanced Permutations — Hard
https://leetcode.com/problems/count-number-of-balanced-permutations/
Best: 方法一：记忆化搜索 + 组合数学
"""

from collections import Counter
from functools import cache
from math import comb

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        @cache
        def dfs(i: int, j: int, a: int, b: int) -> int:
            if i > 9:
                return (j | a | b) == 0
            if a == 0 and j:
                return 0
            ans = 0
            for l in range(min(cnt[i], a) + 1):
                r = cnt[i] - l
                if 0 <= r <= b and l * i <= j:
                    t = comb(a, l) * comb(b, r) * dfs(i + 1, j - l * i, a - l, b - r)
                    ans = (ans + t) % mod
            return ans

        nums = list(map(int, num))
        s = sum(nums)
        if s % 2:
            return 0
        n = len(nums)
        mod = 10**9 + 7
        cnt = Counter(nums)
        return dfs(0, s // 2, n // 2, (n + 1) // 2)

def _test():
    s = Solution()
    assert s.countBalancedPermutations(num='123') == 2
    assert s.countBalancedPermutations(num='112') == 1
    assert s.countBalancedPermutations(num='12345') == 0
    print("LC 3343 OK")

if __name__ == "__main__":
    _test()
