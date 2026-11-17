"""
LC 2277. Closest Node to Path in Tree — Hard
https://leetcode.com/problems/closest-node-to-path-in-tree/
Best: Binary lifting LCA + find path + BFS distances — O((n+q) log n)
"""
from typing import List
from collections import defaultdict, deque


class Solution:
    def closestNode(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        LOG = n.bit_length()
        parent = [[-1] * n for _ in range(LOG)]
        depth = [0] * n

        def dfs(u, p):
            parent[0][u] = p
            for v in g[u]:
                if v != p:
                    depth[v] = depth[u] + 1
                    dfs(v, u)

        dfs(0, -1)
        for k in range(1, LOG):
            for i in range(n):
                if parent[k - 1][i] != -1:
                    parent[k][i] = parent[k - 1][parent[k - 1][i]]

        def lca(u, v):
            if depth[u] < depth[v]:
                u, v = v, u
            diff = depth[u] - depth[v]
            for k in range(LOG):
                if diff >> k & 1:
                    u = parent[k][u]
            if u == v:
                return u
            for k in range(LOG - 1, -1, -1):
                if parent[k][u] != parent[k][v]:
                    u = parent[k][u]
                    v = parent[k][v]
            return parent[0][u]

        def on_path(a, b, c):
            # node x is on path a-b iff dist(a,x)+dist(x,b)==dist(a,b)
            def dist(x, y):
                return depth[x] + depth[y] - 2 * depth[lca(x, y)]
            return dist(a, c) + dist(c, b) == dist(a, b)

        ans = []
        for start, end, node in query:
            # BFS from node, first hit path start-end
            q = deque([node])
            seen = {node}
            found = node
            while q:
                u = q.popleft()
                if on_path(start, end, u):
                    found = u
                    break
                for v in g[u]:
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
            ans.append(found)
        return ans


def _test():
    s = Solution()
    assert s.closestNode(7, [[0,1],[0,2],[0,3],[1,4],[2,5],[2,6]], [[5,3,4],[5,3,6]]) == [0, 2]
    assert s.closestNode(3, [[0,1],[1,2]], [[0,1,2]]) == [1]
    print("LC 2277 OK")


if __name__ == "__main__":
    _test()
