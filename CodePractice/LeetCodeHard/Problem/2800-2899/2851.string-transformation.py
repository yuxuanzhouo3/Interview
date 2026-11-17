"""
LC 2851. String Transformation — Hard
https://leetcode.com/problems/string-transformation/
Best: KMP + matrix/closed-form rotate DP — O(n + log k)
"""


class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        ss = s + s

        def kmp_find(text: str, pat: str):
            m = len(pat)
            lps = [0] * m
            length = 0
            i = 1
            while i < m:
                if pat[i] == pat[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                elif length:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
            res = []
            i = j = 0
            while i < len(text):
                if text[i] == pat[j]:
                    i += 1
                    j += 1
                    if j == m:
                        res.append(i - j)
                        j = lps[j - 1]
                elif j:
                    j = lps[j - 1]
                else:
                    i += 1
            return res

        starts = kmp_find(ss[:-1], t)
        if not starts:
            return 0
        ninv = pow(n, MOD - 2, MOD)
        nm1 = n - 1
        pk = pow(nm1, k, MOD)
        m1k = 1 if k % 2 == 0 else MOD - 1
        same = (pk + nm1 * m1k) % MOD * ninv % MOD
        other = (pk - m1k) % MOD * ninv % MOD
        ans = 0
        for p in starts:
            ans = (ans + (same if p == 0 else other)) % MOD
        return ans


def _test():
    s = Solution()
    assert s.numberOfWays("abcd", "cdab", 2) == 2
    assert s.numberOfWays("ababab", "ababab", 1) == 2
    print("LC 2851 OK")


if __name__ == "__main__":
    _test()
