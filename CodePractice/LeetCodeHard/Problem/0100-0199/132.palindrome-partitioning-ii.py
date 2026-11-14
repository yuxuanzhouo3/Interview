"""
LC 132. Palindrome Partitioning II — Hard
Best: DP cuts + palindrome preprocess — O(n^2)
"""

class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        is_pal = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or is_pal[i + 1][j - 1]):
                    is_pal[i][j] = True
        cuts = list(range(n))
        for i in range(n):
            if is_pal[0][i]:
                cuts[i] = 0
                continue
            for j in range(i):
                if is_pal[j + 1][i]:
                    cuts[i] = min(cuts[i], cuts[j] + 1)
        return cuts[-1]


def _test():
    s = Solution()
    assert s.minCut("aab") == 1
    assert s.minCut("a") == 0
    assert s.minCut("ab") == 1
    print("LC 132 OK")


if __name__ == "__main__":
    _test()
