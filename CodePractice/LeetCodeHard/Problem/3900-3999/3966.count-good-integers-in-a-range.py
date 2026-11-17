"""
LC 3966. Count Good Integers in a Range — Hard
https://leetcode.com/problems/count-good-integers-in-a-range/
Best: digit DP — adjacent digits differ by at most k
"""
from functools import lru_cache


class Solution:
    def goodIntegers(self, l: int, r: int, k: int) -> int:
        def count_to(n: int) -> int:
            if n < 0:
                return 0
            s = str(n)

            @lru_cache(None)
            def dp(pos: int, prev: int, tight: bool, started: bool) -> int:
                if pos == len(s):
                    return 1 if started else 0
                up = int(s[pos]) if tight else 9
                ans = 0
                for d in range(0, up + 1):
                    nt = tight and d == up
                    if not started:
                        if d == 0:
                            ans += dp(pos + 1, -1, nt, False)
                        else:
                            ans += dp(pos + 1, d, nt, True)
                    else:
                        if abs(d - prev) <= k:
                            ans += dp(pos + 1, d, nt, True)
                return ans

            return dp(0, -1, True, False)

        return count_to(r) - count_to(l - 1)

def _test():
    s = Solution()
    assert s.goodIntegers(l=10, r=15, k=1) == 3
    assert s.goodIntegers(l=201, r=204, k=2) == 2
    print("LC 3966 OK")


if __name__ == "__main__":
    _test()
