"""
LC 1392. Longest Happy Prefix — Hard
https://leetcode.com/problems/longest-happy-prefix/
Best: KMP LPS — O(n)
"""


class Solution:
    def longestPrefix(self, s: str) -> str:
        n = len(s)
        lps = [0] * n
        length = 0
        i = 1
        while i < n:
            if s[i] == s[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        return s[: lps[-1]]


def _test():
    s = Solution()
    assert s.longestPrefix("level") == "l"
    assert s.longestPrefix("ababab") == "abab"
    assert s.longestPrefix("leetcodeleet") == "leet"
    assert s.longestPrefix("a") == ""
    print("LC 1392 OK")


if __name__ == "__main__":
    _test()
