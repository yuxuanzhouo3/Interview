"""
LC 2213. Longest Substring of One Repeating Character — Hard
https://leetcode.com/problems/longest-substring-of-one-repeating-character/
Best: 线段树维护 pref/suf/best — O((n+k) log n)
"""

from typing import List


class Solution:
    def longestRepeating(self, s: str, queryCharacters: str, queryIndices: List[int]) -> List[int]:
        ca = list(s)
        n = len(ca)

        class Node:
            __slots__ = ("lo", "hi", "best", "pref", "suf", "left", "right")

            def __init__(self, lo, hi):
                self.lo, self.hi = lo, hi
                self.best = self.pref = self.suf = 1
                self.left = self.right = None

        def build(lo, hi):
            node = Node(lo, hi)
            if hi - lo == 1:
                return node
            mid = (lo + hi) // 2
            node.left = build(lo, mid)
            node.right = build(mid, hi)
            merge(node)
            return node

        def merge(node):
            L, R = node.left, node.right
            length = node.hi - node.lo
            mid_len_l = L.hi - L.lo
            mid_len_r = R.hi - R.lo
            node.best = max(L.best, R.best)
            node.pref = L.pref
            node.suf = R.suf
            if ca[L.hi - 1] == ca[R.lo]:
                node.best = max(node.best, L.suf + R.pref)
                if L.pref == mid_len_l:
                    node.pref = L.pref + R.pref
                if R.suf == mid_len_r:
                    node.suf = R.suf + L.suf

        def update(node, idx):
            if node.hi - node.lo == 1:
                return
            if idx < node.left.hi:
                update(node.left, idx)
            else:
                update(node.right, idx)
            merge(node)

        root = build(0, n)
        ans = []
        for ch, idx in zip(queryCharacters, queryIndices):
            ca[idx] = ch
            update(root, idx)
            ans.append(root.best)
        return ans


def _test():
    s = Solution()
    assert s.longestRepeating("babacc", "bcb", [1, 3, 3]) == [3, 3, 4]
    assert s.longestRepeating("abyzz", "aa", [2, 1]) == [2, 3]
    print("LC 2213 OK")


if __name__ == "__main__":
    _test()
