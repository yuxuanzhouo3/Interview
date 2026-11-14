"""
LC 847. Shortest Path Visiting All Nodes — Hard
Best: BFS state (node, mask) — O(n^2 * 2^n)
"""

from typing import List
from collections import deque


class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        full = (1 << n) - 1
        q = deque()
        seen = set()
        for i in range(n):
            q.append((i, 1 << i, 0))
            seen.add((i, 1 << i))
        while q:
            u, mask, d = q.popleft()
            if mask == full:
                return d
            for v in graph[u]:
                nmask = mask | (1 << v)
                if (v, nmask) not in seen:
                    seen.add((v, nmask))
                    q.append((v, nmask, d + 1))
        return -1


def _test():
    s = Solution()
    assert s.shortestPathLength([[1, 2, 3], [0], [0], [0]]) == 4
    assert s.shortestPathLength([[1], [0, 2, 4], [1, 3, 4], [2], [1, 2]]) == 4
    print("LC 847 OK")


if __name__ == "__main__":
    _test()
