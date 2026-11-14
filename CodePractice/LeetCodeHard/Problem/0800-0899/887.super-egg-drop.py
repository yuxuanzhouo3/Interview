"""
LC 887. Super Egg Drop — Hard
Best: DP binary search moves — O(k log n * log) / O(kn) classic
"""

from functools import cache


class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        @cache
        def dp(eggs, floors):
            if floors <= 1 or eggs == 1:
                return floors
            lo, hi = 1, floors
            # find best x minimizing max(break, not break)
            while lo < hi:
                mid = (lo + hi) // 2
                broken = dp(eggs - 1, mid - 1)
                not_broken = dp(eggs, floors - mid)
                if broken < not_broken:
                    lo = mid + 1
                else:
                    hi = mid
            return 1 + max(dp(eggs - 1, lo - 1), dp(eggs, floors - lo))

        return dp(k, n)


def _test():
    s = Solution()
    assert s.superEggDrop(1, 2) == 2
    assert s.superEggDrop(2, 6) == 3
    assert s.superEggDrop(3, 14) == 4
    print("LC 887 OK")


if __name__ == "__main__":
    _test()
