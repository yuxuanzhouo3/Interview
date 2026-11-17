"""
LC 2604. Minimum Time to Eat All Grains — Hard
Best: Sort + binary search + two pointers — O((n+m) log U)
"""

from typing import List


class Solution:
    def minimumTime(self, hens: List[int], grains: List[int]) -> int:
        hens = sorted(hens)
        grains = sorted(grains)
        m = len(grains)

        def check(t):
            j = 0
            for x in hens:
                if j == m:
                    return True
                y = grains[j]
                if y <= x:
                    d = x - y
                    if d > t:
                        return False
                    while j < m and grains[j] <= x:
                        j += 1
                    while j < m and min(d, grains[j] - x) + grains[j] - y <= t:
                        j += 1
                else:
                    while j < m and grains[j] - x <= t:
                        j += 1
            return j == m

        lo, hi = 0, abs(hens[0] - grains[0]) + grains[-1] - grains[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if check(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.minimumTime([3, 6, 7], [2, 4, 7, 9]) == 2
    assert s.minimumTime([4, 6, 109, 111, 213, 215], [5, 110, 214]) == 1
    print("LC 2604 OK")


if __name__ == "__main__":
    _test()
