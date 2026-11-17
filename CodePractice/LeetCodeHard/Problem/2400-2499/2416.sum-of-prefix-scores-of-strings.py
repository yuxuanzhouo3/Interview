"""
LC 2416. Sum of Prefix Scores of Strings — Hard
https://leetcode.com/problems/sum-of-prefix-scores-of-strings/
Best: Trie with prefix counts — O(total length)
"""

from typing import List


class TrieNode:
    __slots__ = ("cnt", "ch")

    def __init__(self):
        self.cnt = 0
        self.ch = {}


class Solution:
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        root = TrieNode()
        for w in words:
            cur = root
            for c in w:
                if c not in cur.ch:
                    cur.ch[c] = TrieNode()
                cur = cur.ch[c]
                cur.cnt += 1
        ans = []
        for w in words:
            cur = root
            s = 0
            for c in w:
                cur = cur.ch[c]
                s += cur.cnt
            ans.append(s)
        return ans


def _test():
    s = Solution()
    assert s.sumPrefixScores(["abc", "ab", "bc", "b"]) == [5, 4, 3, 2]
    assert s.sumPrefixScores(["abcd"]) == [4]
    print("LC 2416 OK")


if __name__ == "__main__":
    _test()
