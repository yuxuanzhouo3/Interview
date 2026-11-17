"""
LC 2234. Maximum Total Beauty of the Gardens — Hard
https://leetcode.com/problems/maximum-total-beauty-of-the-gardens/
Best: Enumerate full gardens + binary search partial — O(n log n)
"""

from bisect import bisect_left
from itertools import accumulate
from typing import List

class Solution:
    def maximumBeauty(
        self, flowers: List[int], newFlowers: int, target: int, full: int, partial: int
    ) -> int:
        flowers = sorted(flowers)
        n = len(flowers)
        s = list(accumulate(flowers, initial=0))
        ans = 0
        i = n - bisect_left(flowers, target)
        for x in range(i, n + 1):
            newFlowers -= 0 if x == 0 else max(target - flowers[n - x], 0)
            if newFlowers < 0:
                break
            l, r = 0, n - x - 1
            while l < r:
                mid = (l + r + 1) >> 1
                if flowers[mid] * (mid + 1) - s[mid + 1] <= newFlowers:
                    l = mid
                else:
                    r = mid - 1
            y = 0
            if r != -1:
                cost = flowers[l] * (l + 1) - s[l + 1]
                y = min(flowers[l] + (newFlowers - cost) // (l + 1), target - 1)
            ans = max(ans, x * full + y * partial)
        return ans

def _test():

    s = Solution()
    assert s.maximumBeauty([1, 3, 1, 1], 7, 6, 12, 1) == 14
    assert s.maximumBeauty([2, 4, 5, 3], 10, 5, 2, 6) == 30
    print("LC 2234 OK")

if __name__ == "__main__":
    _test()
