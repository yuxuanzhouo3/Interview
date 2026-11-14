"""
LC 2050. Parallel Courses III — Hard
https://leetcode.com/problems/parallel-courses-iii/
Best: topo DP — O(n+m)
"""

from typing import List
from collections import deque


class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        g = [[] for _ in range(n)]
        indeg = [0] * n
        for a, b in relations:
            g[a - 1].append(b - 1)
            indeg[b - 1] += 1
        finish = time[:]
        q = deque(i for i in range(n) if indeg[i] == 0)
        while q:
            u = q.popleft()
            for v in g[u]:
                finish[v] = max(finish[v], finish[u] + time[v])
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return max(finish)


def _test():
    s = Solution()
    assert s.minimumTime(3, [[1, 3], [2, 3]], [3, 2, 5]) == 8
    assert s.minimumTime(5, [[1, 5], [2, 5], [3, 5], [3, 4], [4, 5]], [1, 2, 3, 4, 5]) == 12
    print("LC 2050 OK")


if __name__ == "__main__":
    _test()
