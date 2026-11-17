"""
LC 1316. Distinct Echo Substrings — Hard
https://leetcode.com/problems/distinct-echo-substrings/
Best: Rolling hash — O(n^2)
"""


class Solution:
    def distinctEchoSubstrings(self, text: str) -> int:
        n = len(text)
        MOD = 10**9 + 7
        BASE = 31
        h = [0] * (n + 1)
        p = [1] * (n + 1)
        for i, ch in enumerate(text):
            h[i + 1] = (h[i] * BASE + (ord(ch) - 96)) % MOD
            p[i + 1] = p[i] * BASE % MOD

        def get(l, r):  # [l, r)
            return (h[r] - h[l] * p[r - l]) % MOD

        seen = set()
        for length in range(1, n // 2 + 1):
            for i in range(n - 2 * length + 1):
                if get(i, i + length) == get(i + length, i + 2 * length):
                    seen.add(get(i, i + 2 * length))
        return len(seen)


def _test():
    s = Solution()
    assert s.distinctEchoSubstrings("abcabcabc") == 3
    assert s.distinctEchoSubstrings("leetcodeleetcode") == 2
    print("LC 1316 OK")


if __name__ == "__main__":
    _test()
