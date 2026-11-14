"""
LC 140. Word Break II — Hard
Best: Memo DFS collect sentences — O(2^n * n) worst
"""

from typing import List
from functools import cache


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        words = set(wordDict)

        @cache
        def dfs(i: int):
            if i == len(s):
                return [""]
            res = []
            for j in range(i + 1, len(s) + 1):
                w = s[i:j]
                if w in words:
                    for tail in dfs(j):
                        res.append(w if not tail else w + " " + tail)
            return res

        return dfs(0)


def _test():
    s = Solution()
    r = s.wordBreak("catsanddog", ["cat", "cats", "and", "sand", "dog"])
    assert sorted(r) == sorted(["cats and dog", "cat sand dog"])
    print("LC 140 OK")


if __name__ == "__main__":
    _test()
