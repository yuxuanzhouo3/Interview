"""
LC 3093. Longest Common Suffix Queries — Hard
https://leetcode.com/problems/longest-common-suffix-queries/
Best: Reverse trie storing shortest word index — O(total length)
"""

from typing import List


class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        class Node:
            __slots__ = ("ch", "best")
            def __init__(self):
                self.ch = {}
                self.best = -1  # index of best word through this node

        def better(i, j, words):
            if i < 0:
                return j
            if j < 0:
                return i
            if len(words[i]) != len(words[j]):
                return i if len(words[i]) < len(words[j]) else j
            return i if i < j else j

        root = Node()
        words = wordsContainer
        # global best for empty suffix
        global_best = 0
        for i, w in enumerate(words):
            if len(w) < len(words[global_best]) or (len(w) == len(words[global_best]) and i < global_best):
                global_best = i
            node = root
            node.best = better(node.best, i, words)
            for c in reversed(w):
                if c not in node.ch:
                    node.ch[c] = Node()
                node = node.ch[c]
                node.best = better(node.best, i, words)

        ans = []
        for q in wordsQuery:
            node = root
            best = node.best
            for c in reversed(q):
                if c not in node.ch:
                    break
                node = node.ch[c]
                best = node.best
            ans.append(best)
        return ans


def _test():
    s = Solution()
    assert s.stringIndices(["abcd", "bcd", "xbcd"], ["cd", "bcd", "xyz"]) == [1, 1, 1]
    assert s.stringIndices(["abcdefgh", "poiuygh", "ghghgh"], ["gh", "acbfgh", "acbfegh"]) == [2, 0, 2]
    print("LC 3093 OK")


if __name__ == "__main__":
    _test()
