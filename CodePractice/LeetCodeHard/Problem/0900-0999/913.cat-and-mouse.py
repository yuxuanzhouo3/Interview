"""
LC 913. Cat and Mouse — Hard
https://leetcode.com/problems/cat-and-mouse/
Best: DP on (mouse,cat,turn) + degree topo — O(n^3)
"""

from typing import List
from collections import deque


class Solution:
    def catMouseGame(self, graph: List[List[int]]) -> int:
        n = len(graph)
        # result[m][c][t]: 0 draw, 1 mouse win, 2 cat win; t=0 mouse turn, 1 cat turn
        DRAW, MOUSE, CAT = 0, 1, 2
        color = [[[DRAW] * 2 for _ in range(n)] for _ in range(n)]
        degree = [[[0] * 2 for _ in range(n)] for _ in range(n)]
        for m in range(n):
            for c in range(n):
                degree[m][c][0] = len(graph[m])
                degree[m][c][1] = len(graph[c])
                # cat cannot go to 0
                if 0 in graph[c]:
                    degree[m][c][1] -= 1

        q = deque()
        for c in range(n):
            for t in range(2):
                color[0][c][t] = MOUSE
                q.append((0, c, t, MOUSE))
                if c > 0:
                    color[c][c][t] = CAT
                    q.append((c, c, t, CAT))

        def parents(m, c, t):
            if t == 0:  # mouse just moved → previous was cat turn
                for c2 in graph[c]:
                    if c2 == 0:
                        continue
                    yield m, c2, 1
            else:
                for m2 in graph[m]:
                    yield m2, c, 0

        while q:
            m, c, t, res = q.popleft()
            for m2, c2, t2 in parents(m, c, t):
                if color[m2][c2][t2] != DRAW:
                    continue
                # if previous player can force a win
                if (t2 == 0 and res == MOUSE) or (t2 == 1 and res == CAT):
                    color[m2][c2][t2] = res
                    q.append((m2, c2, t2, res))
                else:
                    degree[m2][c2][t2] -= 1
                    if degree[m2][c2][t2] == 0:
                        color[m2][c2][t2] = res
                        q.append((m2, c2, t2, res))
        return color[1][2][0]


def _test():
    s = Solution()
    assert s.catMouseGame([[2, 5], [3], [0, 4, 5], [1, 4, 5], [2, 3], [0, 2, 3]]) == 0
    assert s.catMouseGame([[1, 3], [0], [3], [0, 2]]) == 1
    print("LC 913 OK")


if __name__ == "__main__":
    _test()
