"""
LC 2663. Lexicographically Smallest Beautiful String — Hard
https://leetcode.com/problems/lexicographically-smallest-beautiful-string/
Best: Greedy next permutation-like — O(n|Σ|)
"""

class Solution:
    def smallestBeautifulString(self, s: str, k: int) -> str:
        n = len(s)
        a = [ord(c) - 97 for c in s]

        def ok(i, ch):
            if i >= 1 and a[i - 1] == ch:
                return False
            if i >= 2 and a[i - 2] == ch:
                return False
            return True

        for i in range(n - 1, -1, -1):
            for ch in range(a[i] + 1, k):
                if ok(i, ch):
                    a[i] = ch
                    for j in range(i + 1, n):
                        for c2 in range(k):
                            if ok(j, c2):
                                a[j] = c2
                                break
                    return ''.join(chr(x + 97) for x in a)
            # reset for backtrack conceptually
        return ""


def _test():
    s = Solution()
    assert s.smallestBeautifulString("abcz", 26) == "abda"
    assert s.smallestBeautifulString("dc", 4) == ""
    print("LC 2663 OK")


if __name__ == "__main__":
    _test()

