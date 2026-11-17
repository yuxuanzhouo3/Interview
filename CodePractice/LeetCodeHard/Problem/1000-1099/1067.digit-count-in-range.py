"""
LC 1067. Digit Count in Range — Hard
https://leetcode.com/problems/digit-count-in-range/
Tags: Math, Digit DP
Best: 数位 DP 统计 digit 出现次数
"""

class Solution:
    def digitsCount(self, d: int, low: int, high: int) -> int:
        def count(n: int) -> int:
            if n < 0:
                return 0
            s = str(n)
            m = len(s)
            from functools import lru_cache

            @lru_cache(None)
            def dp(pos, tight, lead, cnt):
                if pos == m:
                    return cnt
                up = int(s[pos]) if tight else 9
                ans = 0
                for dig in range(0, up + 1):
                    nt = tight and dig == up
                    nl = lead and dig == 0
                    nc = cnt
                    if not nl and dig == d:
                        nc += 1
                    # special: d==0 and leading zeros shouldn't count
                    if d == 0 and nl:
                        nc = cnt
                    ans += dp(pos + 1, nt, nl, nc)
                return ans

            return dp(0, True, True, 0)

        return count(high) - count(low - 1)


def _test():
    sol = Solution()
    assert sol.digitsCount(1, 1, 13) == 6
    assert sol.digitsCount(0, 1, 10) == 1
    print("LC 1067 OK")


if __name__ == "__main__":
    _test()

