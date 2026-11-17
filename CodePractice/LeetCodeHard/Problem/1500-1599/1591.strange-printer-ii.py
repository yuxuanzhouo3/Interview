"""
LC 1591. Strange Printer II — Hard
https://leetcode.com/problems/strange-printer-ii/
Best: Bounding box + topological sort — O(C · mn + C^2)
"""

from collections import defaultdict, deque
from typing import List


class Solution:
    def isPrintable(self, targetGrid: List[List[int]]) -> bool:
        m, n = len(targetGrid), len(targetGrid[0])
        colors = set()
        for row in targetGrid:
            colors.update(row)
        bound = {}
        for c in colors:
            r1 = c1 = 60
            r2 = c2 = -1
            for i in range(m):
                for j in range(n):
                    if targetGrid[i][j] == c:
                        r1, r2 = min(r1, i), max(r2, i)
                        c1, c2 = min(c1, j), max(c2, j)
            bound[c] = (r1, r2, c1, c2)

        graph = defaultdict(set)
        indeg = {c: 0 for c in colors}
        for c, (r1, r2, c1, c2) in bound.items():
            for i in range(r1, r2 + 1):
                for j in range(c1, c2 + 1):
                    other = targetGrid[i][j]
                    if other != c and other not in graph[c]:
                        graph[c].add(other)
                        indeg[other] += 1

        q = deque([c for c in colors if indeg[c] == 0])
        seen = 0
        while q:
            u = q.popleft()
            seen += 1
            for v in graph[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return seen == len(colors)


def _test():
    s = Solution()
    assert s.isPrintable([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]) is True
    assert s.isPrintable([[1, 1, 1, 1], [1, 1, 3, 3], [1, 1, 3, 4], [5, 5, 1, 4]]) is True
    assert s.isPrintable([[1, 2, 1], [2, 1, 2], [1, 2, 1]]) is False
    print("LC 1591 OK")


if __name__ == "__main__":
    _test()
