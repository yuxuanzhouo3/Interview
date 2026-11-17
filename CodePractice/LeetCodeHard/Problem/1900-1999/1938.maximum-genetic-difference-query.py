"""
LC 1938. Maximum Genetic Difference Query — Hard
https://leetcode.com/problems/maximum-genetic-difference-query/
Best: DFS + binary XOR trie — O((n+q) · 18)
"""

from collections import defaultdict
from typing import List

class Trie:
    __slots__ = 'ch', 'cnt'

    def __init__(self):
        self.ch = [None, None]
        self.cnt = 0

    def insert(self, x: int, d: int = 1):
        node = self
        for b in range(17, -1, -1):
            bit = (x >> b) & 1
            if node.ch[bit] is None:
                node.ch[bit] = Trie()
            node = node.ch[bit]
            node.cnt += d

    def query(self, x: int) -> int:
        node = self
        ans = 0
        for b in range(17, -1, -1):
            bit = (x >> b) & 1
            want = 1 - bit
            if node.ch[want] and node.ch[want].cnt > 0:
                ans |= 1 << b
                node = node.ch[want]
            else:
                node = node.ch[bit]
        return ans

class Solution:
    def maxGeneticDifference(self, parents: List[int], queries: List[List[int]]) -> List[int]:
        n = len(parents)
        g = defaultdict(list)
        root = 0
        for i, p in enumerate(parents):
            if p == -1:
                root = i
            else:
                g[p].append(i)
        qs = defaultdict(list)
        for i, (node, val) in enumerate(queries):
            qs[node].append((i, val))
        ans = [0] * len(queries)
        trie = Trie()

        def dfs(u: int):
            trie.insert(u, 1)
            for i, val in qs[u]:
                ans[i] = trie.query(val)
            for v in g[u]:
                dfs(v)
            trie.insert(u, -1)

        dfs(root)
        return ans

def _test():

    s = Solution()
    assert s.maxGeneticDifference([-1, 0, 1, 1], [[0, 2], [3, 2], [2, 5]]) == [2, 3, 7]
    assert s.maxGeneticDifference([3, 7, -1, 2, 0, 7, 0, 2], [[4, 6], [1, 15], [0, 5]]) == [6, 14, 7]
    print("LC 1938 OK")

if __name__ == "__main__":
    _test()
