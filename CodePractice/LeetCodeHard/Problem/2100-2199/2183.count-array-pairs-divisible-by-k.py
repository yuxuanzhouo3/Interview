"""
LC 2183. Count Array Pairs Divisible by K — Hard
https://leetcode.com/problems/count-array-pairs-divisible-by-k/
Best: gcd count: pairs with gcd(a,k)*gcd(b,k) %k ==0 — O(n√k)
"""


from typing import List
from math import gcd
from collections import Counter


class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        # count by gcd(num, k)
        cnt = Counter()
        for x in nums:
            cnt[gcd(x, k)] += 1
        gcds = list(cnt.keys())
        ans = 0
        for i, g1 in enumerate(gcds):
            for j, g2 in enumerate(gcds):
                if g1 * g2 % k == 0:
                    if i < j:
                        ans += cnt[g1] * cnt[g2]
                    elif i == j:
                        ans += cnt[g1] * (cnt[g1] - 1) // 2
        return ans


def _test():
    s = Solution()
    assert s.countPairs([1, 2, 3, 4, 5], 2) == 7
    assert s.countPairs([1, 2, 3, 4], 5) == 0
    print("LC 2183 OK")


if __name__ == "__main__":
    _test()

