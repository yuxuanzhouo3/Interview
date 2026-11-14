"""
LC 472. Concatenated Words — Hard
https://leetcode.com/problems/concatenated-words/
Best: Sort by length + DP word-break — O(N * L^2)
"""

from typing import List


class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        word_set = set(words)
        memo = {}

        def can_form(w: str) -> bool:
            if w in memo:
                return memo[w]
            for i in range(1, len(w)):
                prefix, suffix = w[:i], w[i:]
                if prefix in word_set and (suffix in word_set or can_form(suffix)):
                    memo[w] = True
                    return True
            memo[w] = False
            return False

        return [w for w in words if w and can_form(w)]


def _test():
    s = Solution()
    words = [
        "cat",
        "cats",
        "catsdogcats",
        "dog",
        "dogcatsdog",
        "hippopotamuses",
        "rat",
        "ratcatdogcat",
    ]
    got = set(s.findAllConcatenatedWordsInADict(words))
    assert got == {"catsdogcats", "dogcatsdog", "ratcatdogcat"}
    assert s.findAllConcatenatedWordsInADict(["cat", "dog", "catdog"]) == ["catdog"]
    assert s.findAllConcatenatedWordsInADict([]) == []
    print("LC 472 OK")


if __name__ == "__main__":
    _test()
