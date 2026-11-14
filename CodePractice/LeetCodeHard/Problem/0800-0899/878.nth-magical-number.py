"""
LC 878. Nth Magical Number — Hard
Best: Binary search + LCM inclusion — O(log)
"""

import math

MOD = 10**9 + 7


class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        lcm = a // math.gcd(a, b) * b
        lo, hi = 1, n * min(a, b)
        while lo < hi:
            mid = (lo + hi) // 2
            if mid // a + mid // b - mid // lcm >= n:
                hi = mid
            else:
                lo = mid + 1
        return lo % MOD


def _test():
    s = Solution()
    assert s.nthMagicalNumber(1, 2, 3) == 2
    assert s.nthMagicalNumber(4, 2, 3) == 6
    print("LC 878 OK")


if __name__ == "__main__":
    _test()
