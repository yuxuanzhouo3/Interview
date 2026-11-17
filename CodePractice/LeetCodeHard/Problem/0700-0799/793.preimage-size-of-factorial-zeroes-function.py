"""
LC 793. Preimage Size of Factorial Zeroes Function — Hard
Best: Binary search zeta(x)=K — O(log K * log)
"""

class Solution:
    def preimageSizeFZF(self, k: int) -> int:
        def zeta(x):
            ans = 0
            while x:
                x //= 5
                ans += x
            return ans

        def find(k):
            lo, hi = 0, 5 * k + 5
            while lo < hi:
                mid = (lo + hi) // 2
                if zeta(mid) < k:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        return find(k + 1) - find(k)


def _test():
    s = Solution()
    assert s.preimageSizeFZF(0) == 5
    assert s.preimageSizeFZF(5) == 0
    print("LC 793 OK")


if __name__ == "__main__":
    _test()
