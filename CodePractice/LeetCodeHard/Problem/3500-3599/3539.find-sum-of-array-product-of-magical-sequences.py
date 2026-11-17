"""
LC 3539. Find Sum of Array Product of Magical Sequences — Hard
https://leetcode.com/problems/find-sum-of-array-product-of-magical-sequences/
Best: 方法一：组合数学 + 记忆化搜索
"""

from functools import cache
from math import comb

mx = 30
mod = 10**9 + 7
f = [1] + [0] * mx
g = [1] + [0] * mx

for i in range(1, mx + 1):
    f[i] = f[i - 1] * i % mod
    g[i] = pow(f[i], mod - 2, mod)

def comb(m: int, n: int) -> int:
    return f[m] * g[n] * g[m - n] % mod

class Solution:
    def magicalSum(self, m: int, k: int, nums: list[int]) -> int:
        @cache
        def dfs(i: int, j: int, k: int, st: int) -> int:
            if k < 0 or (i == len(nums) and j > 0):
                return 0
            if i == len(nums):
                while st:
                    k -= st & 1
                    st >>= 1
                return int(k == 0)
            res = 0
            for t in range(j + 1):
                nt = t + st
                p = pow(nums[i], t, mod)
                nk = k - (nt & 1)
                res += comb(j, t) * p * dfs(i + 1, j - t, nk, nt >> 1)
                res %= mod
            return res

        ans = dfs(0, m, k, 0)
        dfs.cache_clear()
        return ans

def _test():
    s = Solution()
    assert s.magicalSum(m=5, k=5, nums=[1, 10, 100, 10000, 1000000]) == 991600007
    assert s.magicalSum(m=2, k=2, nums=[5, 4, 3, 2, 1]) == 170
    assert s.magicalSum(m=1, k=1, nums=[28]) == 28
    print("LC 3539 OK")

if __name__ == "__main__":
    _test()
