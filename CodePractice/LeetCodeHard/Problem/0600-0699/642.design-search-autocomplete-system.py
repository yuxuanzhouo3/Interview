"""
LC 642. Design Search Autocomplete System — Hard
https://leetcode.com/problems/design-search-autocomplete-system/
Best: Trie + hot sentences ranking — O(L) per key
"""

from typing import List
from collections import defaultdict


class AutocompleteSystem:
    def __init__(self, sentences: List[str], times: List[int]):
        self.freq = defaultdict(int)
        for s, t in zip(sentences, times):
            self.freq[s] += t
        self.cur = ""

    def input(self, c: str) -> List[str]:
        if c == "#":
            self.freq[self.cur] += 1
            self.cur = ""
            return []
        self.cur += c
        prefix = self.cur
        cands = [s for s in self.freq if s.startswith(prefix)]
        cands.sort(key=lambda s: (-self.freq[s], s))
        return cands[:3]


# LeetCode-style wrapper
class Solution:
    def run(self, sentences, times, inputs):
        ac = AutocompleteSystem(sentences, times)
        return [ac.input(c) for c in inputs]


def _test():
    ac = AutocompleteSystem(["i love you", "island", "iroman", "i love leetcode"], [5, 3, 2, 2])
    assert ac.input("i") == ["i love you", "island", "i love leetcode"]
    assert ac.input(" ") == ["i love you", "i love leetcode"]
    assert ac.input("a") == []
    assert ac.input("#") == []
    assert ac.input("i") == ["i love you", "island", "i love leetcode"]
    print("LC 642 OK")


if __name__ == "__main__":
    _test()
