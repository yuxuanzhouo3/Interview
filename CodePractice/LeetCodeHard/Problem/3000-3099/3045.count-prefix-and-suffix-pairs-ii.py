"""
LC 3045. Count Prefix and Suffix Pairs II — Hard
https://leetcode.com/problems/count-prefix-and-suffix-pairs-ii/
Best: Trie on (prefix,suffix) char pairs — O(total length)
"""

from typing import List


class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        class Node:
            __slots__ = ("child", "cnt")
            def __init__(self):
                self.child = {}
                self.cnt = 0

        root = Node()
        ans = 0
        for w in words:
            node = root
            # query: for each prefix of w that is also suffix, add cnt
            # Actually: count previous words that are prefix AND suffix of w
            # Insert words in order; for current w, walk trie by pairs (w[i], w[n-1-i])
            n = len(w)
            node = root
            for i in range(n):
                key = (w[i], w[n - 1 - i])
                if key not in node.child:
                    node.child[key] = Node()
                node = node.child[key]
                ans += node.cnt
            # mark word end: full word as prefix&suffix of itself for future
            node.cnt += 1
        return ans


def _test():
    s = Solution()
    assert s.countPrefixSuffixPairs(["a", "aba", "ababa", "aa"]) == 4
    assert s.countPrefixSuffixPairs(["pa", "papa", "ma", "mama"]) == 2
    assert s.countPrefixSuffixPairs(["abab", "ab"]) == 0
    print("LC 3045 OK")


if __name__ == "__main__":
    _test()
