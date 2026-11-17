"""
LC 3519. Count Numbers with Non-Decreasing Digits  — Hard
https://leetcode.com/problems/count-numbers-with-non-decreasing-digits/
Tags: Math, String, Dynamic Programming
Best: 最优解
"""
from functools import lru_cache

class Solution:
    def countNumbers(self, l: str, r: str, b: int) -> int:
        MOD = 10**9 + 7

        def to_digits(s: str):
            n = int(s)
            if n == 0:
                return [0]
            ds = []
            while n:
                ds.append(n % b)
                n //= b
            return ds[::-1]

        def count_le(s: str) -> int:
            if int(s) < 0:
                return 0
            digits = to_digits(s)
            m = len(digits)

            @lru_cache(None)
            def dfs(pos, last, tight, started):
                if pos == m:
                    return 1  # include 0 when never started? we count empty as 0 value once
                up = digits[pos] if tight else b - 1
                ans = 0
                for d in range(0, up + 1):
                    nt = tight and (d == up)
                    if not started:
                        if d == 0:
                            ans += dfs(pos + 1, 0, nt, False)
                        else:
                            ans += dfs(pos + 1, d, nt, True)
                    else:
                        if d >= last:
                            ans += dfs(pos + 1, d, nt, True)
                return ans % MOD

            # dfs with not started ending counts 1 for the number 0
            return dfs(0, 0, True, False) % MOD

        # numbers in [l,r] = count_le(r) - count_le(l-1)
        ans_r = count_le(r)
        lv = int(l) - 1
        ans_l = count_le(str(lv)) if lv >= 0 else 0
        return (ans_r - ans_l) % MOD

def _test():
    s = Solution()
    assert s.countNumbers(l="23", r="28", b=8) == 3
    assert s.countNumbers(l="2", r="7", b=2) == 2
    print("LC 3519 OK")

if __name__ == "__main__":
    _test()
