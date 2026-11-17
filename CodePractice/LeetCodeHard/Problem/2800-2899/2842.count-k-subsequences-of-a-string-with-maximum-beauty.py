"""
LC 2842. Count K-Subsequences of a String With Maximum Beauty — Hard
https://leetcode.com/problems/count-k-subsequences-of-a-string-with-maximum-beauty/
Best: Greedy freq + combinatorics — O(n+|Σ|)
"""

from collections import Counter
from math import comb


class Solution:
    def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        cnt = Counter(s)
        if k > len(cnt):
            return 0
        freqs = sorted(cnt.values(), reverse=True)
        # beauty = sum of freqs of chosen distinct chars
        # take top frequencies; if tie at boundary, C(n,r)
        target = sum(freqs[:k])
        # find how we choose
        # group by freq
        from collections import defaultdict
        fc = Counter(freqs)
        need = k
        ans = 1
        for f in sorted(fc.keys(), reverse=True):
            c = fc[f]
            if c <= need:
                ans = ans * pow(f, c, MOD) % MOD
                need -= c
            else:
                # choose `need` out of `c` chars each with freq f
                ans = ans * comb(c, need) % MOD * pow(f, need, MOD) % MOD
                need = 0
            if need == 0:
                break
        return ans


def _test():
    s = Solution()
    assert s.countKSubsequencesWithMaxBeauty("bcca", 2) == 4
    assert s.countKSubsequencesWithMaxBeauty("abbcd", 4) == 2
    print("LC 2842 OK")


if __name__ == "__main__":
    _test()

