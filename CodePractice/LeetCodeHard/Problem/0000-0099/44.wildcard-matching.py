"""
LC 44. Wildcard Matching — Hard
Best: Greedy with star backtrack — O(mn) worst, often faster
"""

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        i = j = 0
        star = -1
        match = 0
        m, n = len(s), len(p)
        while i < m:
            if j < n and p[j] in (s[i], "?"):
                i += 1
                j += 1
            elif j < n and p[j] == "*":
                star = j
                match = i
                j += 1
            elif star != -1:
                j = star + 1
                match += 1
                i = match
            else:
                return False
        while j < n and p[j] == "*":
            j += 1
        return j == n


def _test():
    sol = Solution()
    assert sol.isMatch("aa", "a") is False
    assert sol.isMatch("aa", "*") is True
    assert sol.isMatch("cb", "?a") is False
    assert sol.isMatch("adceb", "*a*b") is True
    print("LC 44 OK")


if __name__ == "__main__":
    _test()
