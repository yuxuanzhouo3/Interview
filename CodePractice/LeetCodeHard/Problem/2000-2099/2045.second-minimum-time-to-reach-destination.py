"""
LC 2045. Second Minimum Time to Reach Destination — Hard
https://leetcode.com/problems/second-minimum-time-to-reach-destination/
Best: BFS track 1st & 2nd arrival — O(n+m)
"""

from typing import List
from collections import deque, defaultdict


class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        g = [[] for _ in range(n + 1)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        dist = [[-1, -1] for _ in range(n + 1)]
        dist[1][0] = 0
        q = deque([(1, 0)])  # node, cur_time
        while q:
            u, t = q.popleft()
            for v in g[u]:
                # wait for green
                wait = 0
                if (t // change) % 2 == 1:
                    wait = change - (t % change)
                nt = t + wait + time
                if dist[v][0] == -1:
                    dist[v][0] = nt
                    q.append((v, nt))
                elif dist[v][1] == -1 and dist[v][0] != nt:
                    dist[v][1] = nt
                    q.append((v, nt))
        return dist[n][1]


def _test():
    s = Solution()
    assert s.secondMinimum(5, [[1, 2], [1, 3], [1, 4], [3, 4], [4, 5]], 3, 5) == 13
    assert s.secondMinimum(2, [[1, 2]], 3, 2) == 11
    print("LC 2045 OK")


if __name__ == "__main__":
    _test()
