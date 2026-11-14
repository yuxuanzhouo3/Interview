"""
LC 214. Shortest Palindrome — Hard
Best: KMP on s + '#' + reverse(s) — O(n)
"""

class Solution:
    def shortestPalindrome(self, s: str) -> str:
        if not s:
            return s
        rev = s[::-1]
        comb = s + "#" + rev
        lps = [0] * len(comb)
        for i in range(1, len(comb)):
            j = lps[i - 1]
            while j > 0 and comb[i] != comb[j]:
                j = lps[j - 1]
            if comb[i] == comb[j]:
                j += 1
            lps[i] = j
        add = rev[: len(s) - lps[-1]]
        return add + s


def _test():
    s = Solution()
    assert s.shortestPalindrome("aacecaaa") == "aaacecaaa"
    assert s.shortestPalindrome("abcd") == "dcbabcd"
    print("LC 214 OK")


if __name__ == "__main__":
    _test()
