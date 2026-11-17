"""
LC 3104. Find Longest Self-Contained Substring — Hard
https://leetcode.com/problems/find-longest-self-contained-substring/
Best: Expand by char first/last — O(26n)
"""


class Solution:
    def maxSubstringLength(self, s: str) -> int:
        n = len(s)
        first = [-1] * 26
        last = [-1] * 26
        for i, ch in enumerate(s):
            c = ord(ch) - 97
            if first[c] < 0:
                first[c] = i
            last[c] = i
        ans = -1
        for c in range(26):
            if first[c] < 0:
                continue
            L, R = first[c], last[c]
            i = L
            ok = True
            while i <= R:
                t = ord(s[i]) - 97
                if first[t] < L:
                    ok = False
                    break
                R = max(R, last[t])
                if R - L + 1 == n:
                    ok = False
                    break
                i += 1
            if ok:
                ans = max(ans, R - L + 1)
        return ans


def _test():
    s = Solution()
    assert s.maxSubstringLength("abba") == 2
    assert s.maxSubstringLength("abab") == -1
    assert s.maxSubstringLength("abacd") == 3
    print("LC 3104 OK")


if __name__ == "__main__":
    _test()

