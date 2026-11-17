"""
LC 2376. Count Special Integers — Hard
https://leetcode.com/problems/count-special-integers/
Best: Digit DP unique digits — O(log n · 2^10 · 10)
"""


class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)
        m = len(s)

        from functools import lru_cache

        @lru_cache(None)
        def dp(pos: int, mask: int, tight: bool, started: bool) -> int:
            if pos == m:
                return 1 if started else 0
            up = int(s[pos]) if tight else 9
            ans = 0
            for d in range(up + 1):
                nt = tight and d == up
                if not started:
                    if d == 0:
                        ans += dp(pos + 1, mask, nt, False)
                    else:
                        ans += dp(pos + 1, mask | (1 << d), nt, True)
                else:
                    if mask & (1 << d):
                        continue
                    ans += dp(pos + 1, mask | (1 << d), nt, True)
            return ans

        return dp(0, 0, True, False)


def _test():
    s = Solution()
    assert s.countSpecialNumbers(20) == 19
    assert s.countSpecialNumbers(5) == 5
    assert s.countSpecialNumbers(135) == 110
    print("LC 2376 OK")


if __name__ == "__main__":
    _test()
