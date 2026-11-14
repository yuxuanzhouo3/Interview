"""
LC 745. Prefix and Suffix Search — Hard
Best: Trie of word#word / map all prefix-suffix keys — O(N L^2)
"""

from typing import List


class WordFilter:
    def __init__(self, words: List[str]):
        self.mp = {}
        for i, w in enumerate(words):
            for p in range(len(w) + 1):
                for s in range(len(w) + 1):
                    self.mp[w[:p] + "#" + w[s:]] = i

    def f(self, pref: str, suff: str) -> int:
        return self.mp.get(pref + "#" + suff, -1)


def _test():
    wf = WordFilter(["apple"])
    assert wf.f("a", "e") == 0
    print("LC 745 OK")


if __name__ == "__main__":
    _test()
