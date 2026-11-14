"""
LC 87. Scramble String — Hard
Best: Memo DFS / interval DP — O(n^4)
"""

from functools import cache


class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        @cache
        def dfs(a: str, b: str) -> bool:
            if a == b:
                return True
            if sorted(a) != sorted(b):
                return False
            n = len(a)
            for i in range(1, n):
                if dfs(a[:i], b[:i]) and dfs(a[i:], b[i:]):
                    return True
                if dfs(a[:i], b[-i:]) and dfs(a[i:], b[:-i]):
                    return True
            return False

        return dfs(s1, s2)


def _test():
    s = Solution()
    assert s.isScramble("great", "rgeat") is True
    assert s.isScramble("abcde", "caebd") is False
    assert s.isScramble("a", "a") is True
    print("LC 87 OK")


if __name__ == "__main__":
    _test()
