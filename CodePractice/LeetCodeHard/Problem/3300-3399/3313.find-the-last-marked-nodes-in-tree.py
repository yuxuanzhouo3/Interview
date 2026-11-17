"""
LC 3313. Find the Last Marked Nodes in Tree — Hard
Best: Tree reroot diameter ends — O(n)
"""

from typing import List
from collections import defaultdict

class Solution:
    def lastMarkedNodes(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        g = defaultdict(list)
        for a,b in edges:
            g[a].append(b); g[b].append(a)
        def bfs(src):
            dist=[-1]*n; dist[src]=0
            q=[src]
            for u in q:
                for v in g[u]:
                    if dist[v]<0:
                        dist[v]=dist[u]+1; q.append(v)
            far=max(range(n), key=lambda i: dist[i])
            return far, dist
        a,_=bfs(0)
        b,db=bfs(a)
        _,da=bfs(b)
        # last marked for each start u is farther of a,b from u (marking spreads from neighbors each second)
        return [a if da[i]>=db[i] else b for i in range(n)]

def _test():
    s=Solution()
    r=s.lastMarkedNodes([[0,1],[0,2]])
    assert len(r)==3
    print("LC 3313", r, "OK")

if __name__ == "__main__":
    _test()
