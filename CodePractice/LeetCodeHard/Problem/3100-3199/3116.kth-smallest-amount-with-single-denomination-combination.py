"""
LC 3116. Kth Smallest Amount With Single Denomination Combination — Hard
https://leetcode.com/problems/kth-smallest-amount-with-single-denomination-combination/
Best: Binary search + inclusion-exclusion LCM — O(2^n log ANS)
"""

from typing import List
from math import lcm


class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)

        def count(x: int) -> int:
            # numbers <= x divisible by at least one coin
            total = 0
            for mask in range(1, 1 << n):
                L = 1
                bits = 0
                ok = True
                for i in range(n):
                    if mask >> i & 1:
                        bits += 1
                        L = lcm(L, coins[i])
                        if L > x:
                            ok = False
                            break
                if not ok:
                    continue
                c = x // L
                total += c if bits & 1 else -c
            return total

        lo, hi = 1, k * min(coins)
        while lo < hi:
            mid = (lo + hi) // 2
            if count(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.findKthSmallest([3, 6, 9], 3) == 9
    assert s.findKthSmallest([5, 2], 7) == 12
    print("LC 3116 OK")


if __name__ == "__main__":
    _test()
