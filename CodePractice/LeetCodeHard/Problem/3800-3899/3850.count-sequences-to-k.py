from functools import cache
from math import gcd
"""
LC 3850. Count Sequences to K — Hard
https://leetcode.com/problems/count-sequences-to-k/
Best: doocs/leetcode reference solution (verified on examples)
"""

class Solution:
    def countSequences(self, nums: list[int], k: int) -> int:
        @cache
        def dfs(i: int, p: int, q: int) -> int:
            if i == n:
                return 1 if p == k and q == 1 else 0
            x = nums[i]
            res = dfs(i + 1, p, q)
            g = gcd(p * x, q)
            res += dfs(i + 1, p * x // g, q // g)
            g = gcd(p, q * x)
            res += dfs(i + 1, p // g, q * x // g)
            return res

        n = len(nums)
        ans = dfs(0, 1, 1)
        dfs.cache_clear()
        return ans

def _test():
    s = Solution()
    assert s.countSequences([2,3,2], 6) == 2
    assert s.countSequences([4,6,3], 2) == 2
    assert s.countSequences([1,5], 1) == 3
    print("LC 3850 OK")


if __name__ == "__main__":
    _test()
