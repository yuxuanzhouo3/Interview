"""
LC 2999. Count the Number of Powerful Integers — Hard
https://leetcode.com/problems/count-the-number-of-powerful-integers/
Best: Enumerate prefix length + digit limit — O(log finish)
"""

from functools import lru_cache


class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count(n: int) -> int:
            if n <= 0:
                return 0
            if any(int(c) > limit for c in s):
                return 0
            t = str(n)
            m = len(s)
            ans = 0
            if int(s) <= n:
                ans += 1
            max_len = len(t)
            for plen in range(1, max_len - m + 1):
                if plen + m < max_len:
                    ans += limit * pow(limit + 1, plen - 1)
                    continue
                pref = t[:plen]

                @lru_cache(None)
                def dfs(i: int, tight: bool) -> int:
                    if i == plen:
                        return int(s <= t[plen:]) if tight else 1
                    up = min(int(pref[i]) if tight else 9, limit)
                    lo = 1 if i == 0 else 0
                    res = 0
                    for d in range(lo, up + 1):
                        res += dfs(i + 1, tight and d == int(pref[i]))
                    return res

                ans += dfs(0, True)
            return ans

        return count(finish) - count(start - 1)


def _test():
    s = Solution()
    assert s.numberOfPowerfulInt(1, 6000, 4, "124") == 5
    assert s.numberOfPowerfulInt(15, 215, 6, "10") == 2
    assert s.numberOfPowerfulInt(1000, 2000, 4, "3000") == 0
    print("LC 2999 OK")


if __name__ == "__main__":
    _test()
