"""
LC 3620. Network Recovery Pathways — Hard
https://leetcode.com/problems/network-recovery-pathways/
Best: doocs/leetcode reference solution (verified on examples)
"""

from math import inf
from heapq import heappush, heappop, heapify


class Solution:
    def findMaxPathScore(
        self, edges: list[list[int]], online: list[bool], k: int
    ) -> int:
        def check(mid: int) -> int:
            dist = [inf] * n
            dist[0] = 0
            pq = [(0, 0)]
            while pq:
                d, u = heappop(pq)
                if d > k:
                    return False
                if u == n - 1:
                    return True
                if dist[u] < d:
                    continue
                for v, w in g[u]:
                    if w < mid:
                        continue
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        heappush(pq, (dist[v], v))
            return False

        n = len(online)
        g = [[] for _ in range(n)]
        l, r = inf, 0
        for (
            u,
            v,
            w,
        ) in edges:
            if not online[u] or not online[v]:
                continue
            g[u].append((v, w))
            l = min(l, w)
            r = max(r, w)

        while l < r:
            mid = (l + r + 1) >> 1
            if check(mid):
                l = mid
            else:
                r = mid - 1
        return l if check(l) else -1

def _test():
    s = Solution()
    assert s.findMaxPathScore([[0,1,5],[1,3,10],[0,2,3],[2,3,4]], [True,True,True,True], 10) == 3
    assert s.findMaxPathScore([[0,1,7],[1,4,5],[0,2,6],[2,3,6],[3,4,2],[2,4,6]], [True,True,True,False,True], 12) == 6
    print("LC 3620 OK")


if __name__ == "__main__":
    _test()
