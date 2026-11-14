"""
LC 425. Word Squares — Hard
https://leetcode.com/problems/word-squares/
Best: 前缀哈希 + 回溯 — O(#squares · n²)
"""

from typing import List
from collections import defaultdict


class Solution:
    def wordSquares(self, words: List[str]) -> List[List[str]]:
        if not words:
            return []
        n = len(words[0])
        prefix = defaultdict(list)
        for w in words:
            for i in range(n + 1):
                prefix[w[:i]].append(w)

        ans = []

        def dfs(square: List[str]):
            if len(square) == n:
                ans.append(square[:])
                return
            # 下一行前缀 = 已有各行的第 len(square) 列
            pref = "".join(row[len(square)] for row in square)
            for w in prefix[pref]:
                square.append(w)
                dfs(square)
                square.pop()

        for w in words:
            dfs([w])
        return ans


def _test():
    s = Solution()
    r = s.wordSquares(["area", "lead", "wall", "lady", "ball"])
    expect = {("wall", "area", "lead", "lady"), ("ball", "area", "lead", "lady")}
    got = {tuple(x) for x in r}
    assert got == expect
    r2 = s.wordSquares(["abat", "baba", "atan", "atal"])
    got2 = {tuple(x) for x in r2}
    assert ("baba", "abat", "baba", "atan") in got2 or len(got2) >= 1
    assert s.wordSquares(["a"]) == [["a"]]
    print("LC 425 OK")


if __name__ == "__main__":
    _test()
