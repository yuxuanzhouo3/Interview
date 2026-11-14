"""
LC 828. Count Unique Characters of All Substrings of a Given String — Hard
Best: Contribution of each char by prev/next same — O(n)
"""

class Solution:
    def uniqueLetterString(self, s: str) -> int:
        n = len(s)
        last = [-1] * 26
        prev = [-1] * n
        for i, c in enumerate(s):
            prev[i] = last[ord(c) - 65]
            last[ord(c) - 65] = i
        nxt = [n] * n
        last = [n] * 26
        for i in range(n - 1, -1, -1):
            c = ord(s[i]) - 65
            nxt[i] = last[c]
            last[c] = i
        ans = 0
        for i in range(n):
            ans += (i - prev[i]) * (nxt[i] - i)
        return ans


def _test():
    s = Solution()
    assert s.uniqueLetterString("ABC") == 10
    assert s.uniqueLetterString("ABA") == 8
    print("LC 828 OK")


if __name__ == "__main__":
    _test()
