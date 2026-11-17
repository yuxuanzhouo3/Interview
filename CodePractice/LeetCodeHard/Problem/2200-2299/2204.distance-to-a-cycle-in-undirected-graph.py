"""
LC 2204. Distance to a Cycle in Undirected Graph — Hard
https://leetcode.com/problems/distance-to-a-cycle-in-undirected-graph/
Best: 剥叶子找环 + 多源 BFS — O(n)
"""

from typing import List
from collections import deque, defaultdict


class Solution:
    def distanceToCycle(self, n: int, edges: List[List[int]]) -> List[int]:
        g = defaultdict(set)
        for a, b in edges:
            g[a].add(b)
            g[b].add(a)
        q = deque(i for i in range(n) if len(g[i]) == 1)
        parent = [0] * n
        order = []
        while q:
            i = q.popleft()
            order.append(i)
            for j in list(g[i]):
                g[j].discard(i)
                parent[i] = j
                if len(g[j]) == 1:
                    q.append(j)
            g[i].clear()
        ans = [0] * n
        for i in reversed(order):
            ans[i] = ans[parent[i]] + 1
        return ans


def _test():
    s = Solution()
    assert s.distanceToCycle(7, [[1, 2], [2, 4], [4, 3], [3, 1], [0, 1], [5, 2], [6, 5]]) == [1, 0, 0, 0, 0, 1, 2]
    assert s.distanceToCycle(3, [[0, 1], [1, 2], [0, 2]]) == [0, 0, 0]
    print("LC 2204 OK")


if __name__ == "__main__":
    _test()
