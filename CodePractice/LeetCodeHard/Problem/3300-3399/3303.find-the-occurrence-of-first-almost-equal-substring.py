"""
LC 3303. Find the Occurrence of First Almost Equal Substring — Hard
https://leetcode.com/problems/find-the-occurrence-of-first-almost-equal-substring/
Best: Z-algorithm on pattern#text and reverse — O(n+m)
"""


class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        def z_func(t: str):
            n = len(t)
            z = [0] * n
            l = r = 0
            for i in range(1, n):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < n and t[z[i]] == t[i + z[i]]:
                    z[i] += 1
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            return z

        m, n = len(pattern), len(s)
        if m > n:
            return -1
        z1 = z_func(pattern + "#" + s)
        z2 = z_func(pattern[::-1] + "#" + s[::-1])
        for i in range(n - m + 1):
            pref = z1[m + 1 + i]
            suf = z2[m + 1 + (n - (i + m))]
            if pref + suf >= m - 1:
                return i
        return -1


def _test():
    s = Solution()
    assert s.minStartingIndex("abcdefg", "bcdffg") == 1
    assert s.minStartingIndex("ababbababa", "bacaba") == 4
    assert s.minStartingIndex("abcd", "dba") == -1
    print("LC 3303 OK")


if __name__ == "__main__":
    _test()
