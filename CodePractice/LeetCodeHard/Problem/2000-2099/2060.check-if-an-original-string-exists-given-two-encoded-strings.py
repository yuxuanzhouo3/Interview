"""
LC 2060. Check if an Original String Exists Given Two Encoded Strings — Hard
https://leetcode.com/problems/check-if-an-original-string-exists-given-two-encoded-strings/
Best: 记忆化 DFS (i,j,diff) — O(n^2 * D)
"""

from functools import lru_cache


class Solution:
    def possiblyEquals(self, s1: str, s2: str) -> bool:
        n1, n2 = len(s1), len(s2)

        @lru_cache(None)
        def dfs(i, j, diff):
            if i == n1 and j == n2:
                return diff == 0
            if i < n1 and s1[i].isdigit():
                num = 0
                for k in range(i, min(i + 3, n1)):
                    if not s1[k].isdigit():
                        break
                    num = num * 10 + int(s1[k])
                    if dfs(k + 1, j, diff + num):
                        return True
                return False
            if j < n2 and s2[j].isdigit():
                num = 0
                for k in range(j, min(j + 3, n2)):
                    if not s2[k].isdigit():
                        break
                    num = num * 10 + int(s2[k])
                    if dfs(i, k + 1, diff - num):
                        return True
                return False
            if diff > 0:
                return j < n2 and not s2[j].isdigit() and dfs(i, j + 1, diff - 1)
            if diff < 0:
                return i < n1 and not s1[i].isdigit() and dfs(i + 1, j, diff + 1)
            if i < n1 and j < n2 and s1[i] == s2[j]:
                return dfs(i + 1, j + 1, 0)
            return False

        return dfs(0, 0, 0)


def _test():
    s = Solution()
    assert s.possiblyEquals("internationalization", "i18n") is True
    assert s.possiblyEquals("l123e", "44") is True
    assert s.possiblyEquals("a5b", "c5b") is False
    print("LC 2060 OK")


if __name__ == "__main__":
    _test()
