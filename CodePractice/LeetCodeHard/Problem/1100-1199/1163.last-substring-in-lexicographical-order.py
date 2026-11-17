"""
LC 1163. Last Substring in Lexicographical Order — Hard
https://leetcode.com/problems/last-substring-in-lexicographical-order/
Best: Two pointers (Duval-like) — O(n)
"""


class Solution:
    def lastSubstring(self, s: str) -> str:
        n = len(s)
        i, j, k = 0, 1, 0
        while j + k < n:
            if s[i + k] == s[j + k]:
                k += 1
                continue
            if s[i + k] < s[j + k]:
                i += k + 1
            else:
                j += k + 1
            if i == j:
                j += 1
            k = 0
        return s[i:]


def _test():
    s = Solution()
    assert s.lastSubstring("abab") == "bab"
    assert s.lastSubstring("leetcode") == "tcode"
    assert s.lastSubstring("zzzz") == "zzzz"
    assert s.lastSubstring("a") == "a"
    print("LC 1163 OK")


if __name__ == "__main__":
    _test()
