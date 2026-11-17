"""
LC 3373. Maximize the Number of Target Nodes After Connecting Trees II — Hard
https://leetcode.com/problems/maximize-the-number-of-target-nodes-after-connecting-trees-ii/
Best: Bipartite coloring / even-odd depth counts — O(n+m)
"""

from typing import List
from collections import defaultdict, deque


class Solution:
    def maxTargetNodes(
        self, edges1: List[List[int]], edges2: List[List[int]]
    ) -> List[int]:
        def build(edges):
            n = len(edges) + 1
            g = [[] for _ in range(n)]
            for a, b in edges:
                g[a].append(b)
                g[b].append(a)
            return g

        def color_counts(g):
            n = len(g)
            color = [-1] * n
            color[0] = 0
            q = deque([0])
            while q:
                u = q.popleft()
                for v in g[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        q.append(v)
            c0 = sum(1 for c in color if c == 0)
            c1 = n - c0
            return color, c0, c1

        g1, g2 = build(edges1), build(edges2)
        col1, a0, a1 = color_counts(g1)
        _, b0, b1 = color_counts(g2)
        best2 = max(b0, b1)  # connect to maximize even-distance targets in tree2
        # target = even distance. In tree1, for node i: same color count + best2
        n1 = len(g1)
        ans = [0] * n1
        for i in range(n1):
            ans[i] = (a0 if col1[i] == 0 else a1) + best2
        return ans


def _test():
    s = Solution()
    assert s.maxTargetNodes(
        [[0, 1], [0, 2], [2, 3], [2, 4]],
        [[0, 1], [0, 2], [0, 3], [2, 7], [1, 4], [4, 5], [4, 6]],
    ) == [8, 7, 7, 8, 8]
    assert s.maxTargetNodes(
        [[0, 1], [0, 2], [0, 3], [0, 4]],
        [[0, 1], [1, 2], [2, 3]],
    ) == [3, 6, 6, 6, 6]
    print("LC 3373 OK")


if __name__ == "__main__":
    _test()
