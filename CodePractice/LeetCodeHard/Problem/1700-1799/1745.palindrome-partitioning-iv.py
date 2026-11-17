"""
LC 1745. Palindrome Partitioning IV — Hard
https://leetcode.com/problems/palindrome-partitioning-iv/
Best: precompute isPal[i][j], enum two cuts — O(n^2)
"""


class Solution:
    def checkPartitioning(self, s: str) -> bool:
        n = len(s)
        is_pal = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 1 or is_pal[i + 1][j - 1]):
                    is_pal[i][j] = True
        for i in range(n - 2):
            if not is_pal[0][i]:
                continue
            for j in range(i + 1, n - 1):
                if is_pal[i + 1][j] and is_pal[j + 1][n - 1]:
                    return True
        return False


def _test():
    s = Solution()
    assert s.checkPartitioning("abcbdd") is True
    assert s.checkPartitioning("bcbddxy") is False
    print("LC 1745 OK")


if __name__ == "__main__":
    _test()
