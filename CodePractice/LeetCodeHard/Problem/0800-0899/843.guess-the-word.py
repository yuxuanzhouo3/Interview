"""
LC 843. Guess the Word — Hard
https://leetcode.com/problems/guess-the-word/
Best: Minimax — guess minimizing worst-case remaining — O(n^2 · 10)
"""

from typing import List, Optional
import collections


class Master:
    def __init__(self, secret: str, words: List[str], allowed: int = 10):
        self.secret = secret
        self.allowed = allowed
        self.calls = 0
        self.found = False
        self.word_set = set(words)

    def guess(self, word: str) -> int:
        self.calls += 1
        if self.calls > self.allowed:
            raise RuntimeError("too many guesses")
        if word not in self.word_set:
            return -1
        m = sum(a == b for a, b in zip(word, self.secret))
        if m == 6:
            self.found = True
        return m


class Solution:
    def findSecretWord(self, words: List[str], master: "Master") -> None:
        def match(a: str, b: str) -> int:
            return sum(x == y for x, y in zip(a, b))

        cand = words[:]
        for _ in range(10):
            if not cand:
                return
            # pick word minimizing max bucket size
            best, best_score = cand[0], float("inf")
            for w in cand:
                buckets = [0] * 7
                for o in cand:
                    buckets[match(w, o)] += 1
                score = max(buckets)
                if score < best_score:
                    best_score = score
                    best = w
            g = master.guess(best)
            if g == 6:
                return
            cand = [w for w in cand if match(w, best) == g]


def _test():
    words = ["acckzz", "ccbazz", "eiowzz", "abcczz"]
    master = Master("acckzz", words, 10)
    Solution().findSecretWord(words, master)
    assert master.found
    words2 = ["hamada", "khaled"]
    master2 = Master("hamada", words2, 10)
    Solution().findSecretWord(words2, master2)
    assert master2.found
    print("LC 843 OK")


if __name__ == "__main__":
    _test()
