"""
LC 1397. Find All Good Strings — Hard
https://leetcode.com/problems/find-all-good-strings/
Best: Digit DP + KMP automaton — O(n · |evil| · Σ)
"""

from functools import lru_cache


class Solution:
    def findGoodStrings(self, n: int, s1: str, s2: str, evil: str) -> int:
        MOD = 10**9 + 7
        m = len(evil)
        # KMP pi
        pi = [0] * m
        j = 0
        for i in range(1, m):
            while j and evil[i] != evil[j]:
                j = pi[j - 1]
            if evil[i] == evil[j]:
                j += 1
            pi[i] = j
        # nxt[state][c] = new state after appending c
        nxt = [[0] * 26 for _ in range(m)]
        for st in range(m):
            for c in range(26):
                ch = chr(ord("a") + c)
                k = st
                while k and evil[k] != ch:
                    k = pi[k - 1]
                if evil[k] == ch:
                    k += 1
                nxt[st][c] = k

        @lru_cache(None)
        def dp(pos: int, state: int, tight1: bool, tight2: bool) -> int:
            if state == m:
                return 0
            if pos == n:
                return 1
            lo = ord(s1[pos]) - 97 if tight1 else 0
            hi = ord(s2[pos]) - 97 if tight2 else 25
            ans = 0
            for c in range(lo, hi + 1):
                ans += dp(
                    pos + 1,
                    nxt[state][c],
                    tight1 and c == lo,
                    tight2 and c == hi,
                )
            return ans % MOD

        return dp(0, 0, True, True)


def _test():
    s = Solution()
    assert s.findGoodStrings(2, "aa", "da", "b") == 51
    assert s.findGoodStrings(8, "leetcode", "leetgoes", "leet") == 0
    assert s.findGoodStrings(2, "gx", "gz", "x") == 2
    print("LC 1397 OK")


if __name__ == "__main__":
    _test()
