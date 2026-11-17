"""
LC 3924. Minimum Threshold Path with Limited Heavy Edges — Hard
https://leetcode.com/problems/minimum-threshold-path-with-limited-heavy-edges/
Best: binary search threshold + UF on light edges + BFS on heavy components
"""
from collections import deque


class Solution:
    def minimumThreshold(self, n: int, edges: list[list[int]], source: int, target: int, k: int) -> int:
        if source == target:
            return 0
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            parent[pa] = parent[pb] = min(pa, pb)

        parent = list(range(n))
        for u, v, _ in edges:
            union(u, v)
        if find(source) != find(target):
            return -1

        edges = sorted(edges, key=lambda e: e[2])
        S = [0]
        for e in edges:
            if S[-1] != e[2]:
                S.append(e[2])

        def ok(m: int) -> bool:
            parent[:] = range(n)
            adj = [[] for _ in range(n)]
            for u, v, w in edges:
                if w <= m:
                    union(u, v)
                else:
                    pu, pv = find(u), find(v)
                    if pu != pv:
                        adj[pu].append(pv)
                        adj[pv].append(pu)
            vis = [False] * n
            q = deque()

            def push(x):
                if not vis[x]:
                    vis[x] = True
                    q.append(x)

            push(find(source))
            for _ in range(k):
                for _ in range(len(q)):
                    u = q.popleft()
                    for v in adj[u]:
                        push(v)
            return vis[find(target)]

        lo, hi = 0, len(S) - 2
        res = S[-1]
        while lo <= hi:
            mid = (lo + hi) // 2
            if ok(S[mid]):
                res = S[mid]
                hi = mid - 1
            else:
                lo = mid + 1
        return res


def _test():
    s = Solution()
    assert s.minimumThreshold(6, [[0,1,5],[1,2,3],[3,4,4],[4,5,1],[1,4,2]], 0, 3, 1) == 4
    assert s.minimumThreshold(6, [[0,1,3],[1,2,4],[3,4,5],[4,5,6]], 0, 4, 1) == -1
    assert s.minimumThreshold(4, [[0,1,2],[1,2,2],[2,3,2],[3,0,2]], 0, 0, 0) == 0
    print("LC 3924 OK")


if __name__ == "__main__":
    _test()
